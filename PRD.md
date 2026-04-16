# InstaLLM PRD and Implementation Plan

## Executive summary

InstaLLM (“installm”) is a command-line tool (CLI) that makes it easy to download one or more open-source LLMs from the Hugging Face Hub and immediately expose them as an HTTP API that looks like the OpenAI API—**local or remote** (configurable bind address)—so existing tooling (especially the **OpenAI Python SDK** and agent frameworks like LangChain**) can use your models with little or no code changes. **`installm up` is oriented toward a daemon-style deployment** suitable for laptops and VMs. This API must support streaming, tool calling, and structured outputs (JSON results), while authentication and metrics are optional. *(Streaming = the server sends tokens/events as they’re generated; tool calling = the model returns a structured “call this function with these arguments” object; structured output = the model returns valid JSON that matches a schema you define.)*

Two existing projects give strong design patterns you should reuse:

* **response.js** shows how to implement the Responses API by internally running everything as a stream of events, then either:

  * returning SSE to the client if `stream=true`, or
  * collecting until completion and returning a single JSON response if `stream=false`.

  It also demonstrates building semantic streaming events with incrementing `sequence_number`.

* **Ollama** demonstrates the practical “OpenAI-compatibility” UX: run a local daemon, point OpenAI clients at `http://localhost:<port>/v1`, and use a dummy API key if needed. Its docs also clearly list which OpenAI-like endpoints and fields are supported (including `/v1/responses`), and highlight that compatibility is often “subset + best effort.”

The key product decision for InstaLLM is to split the system into:

* a **Control Plane** (CLI + local state) that downloads models, chooses runtimes, starts/stops processes, and prints a ready-to-use base URL, and
* a **Data Plane** (API server) that serves OpenAI-compatible endpoints and normalizes “hard parts” (streaming, tool-call framing, structured-output behavior) across backends.

This separation lets you start with a simple local MVP and later add Kubernetes/server modes without rewriting your core logic.

---

## Product definition and non-negotiables

### Core goals

#### One-command deployment

A novice user should be able to run something like:

```bash
installm up --model <hf_model_id> [--model <hf_model_id> ...]
```

…and then immediately use an OpenAI-compatible client by setting the base URL to the server root (typically ending in `/v1`). This is the same “just change base URL” workflow used by many OpenAI-compatible providers, and is explicitly how OpenAI-compatible integrations are expected to be used.

#### Download + setup multiple models

“Setup” means ensuring weights are present locally, then ensuring the runtime can load and serve them. Hugging Face’s recommended way to download repositories programmatically is via its caching system (e.g., `snapshot_download`) and/or the CLI `hf download`, both of which avoid re-downloading unchanged files and can target a specific directory.

#### OpenAI-compatible API surface (minimum viable)

To maximize compatibility with real tooling, your MVP should implement (or proxy faithfully):

* `GET /v1/models` *(model discovery)*
* `POST /v1/chat/completions` *(most compatibility; many libraries primarily rely on this)*
* `POST /v1/responses` *(recommended modern API; semantic streaming events)*
* `POST /v1/embeddings` *(commonly needed for RAG / vector search)*

#### Must-have features for LLM requests (your “non-optional” list)

* Streaming (SSE) for both Chat Completions and Responses.
* Tool calling as an API feature: requests can include tools, and responses can include `tool_calls` in the correct OpenAI-shaped structure.
* Structured outputs: callers can request JSON output, ideally via OpenAI’s `response_format` (Chat Completions) and/or `text.format` (Responses API patterns), with **schema enforcement when the backend supports it**, and **best-effort validate-and-retry** otherwise.

### Who executes tools: you (or LangChain), not the model server

Your intuition is basically correct, and the official docs agree:

* In tool calling, the model does not run tools. It outputs a structured request (“call tool X with args Y”), and the application runtime executes the tool and then feeds results back. Hugging Face’s Transformers docs state this explicitly (“A model cannot actually call the tool itself… it requests a tool call, and it’s your job to handle the call and append it and the result to the chat history”).
* LangChain’s agent runtime similarly executes the tool and records the result as a `ToolMessage`.
* OpenAI’s streaming/tool schemas also emphasize that tool-call arguments may be malformed and should be validated in your code before executing.

So, InstaLLM does **not** need to execute tools to satisfy “tool calling support.” It must support the protocol:

* accept tools,
* emit `tool_calls`, and
* accept tool results when the client sends them as tool role messages (Chat Completions) or as tool output items (Responses).

Server-side tool orchestration can be a future “optional” feature (`response.js` does show one approach via MCP), but it is not required for LangChain-style tool execution.

### Optional features (explicitly optional per you)

* Authentication (API keys, maybe TLS termination).
* Metrics/observability (Prometheus metrics endpoint, tracing).

You should still design “attachment points” (middleware/hooks) so these can be added later without breaking the core CLI or request/response schemas.

### Baseline product decisions (locked)

These choices set defaults and scope for implementation (formerly captured as informal “open questions”; consolidated here).

* **Networking:** Support **local and remote** use from day one: configurable **bind address** (including `0.0.0.0` for VMs and servers), not localhost-only.
* **Process model:** **`installm up` targets a daemon-style deployment** (service keeps running after the terminal closes), suitable for laptops and remote VMs. The exact mechanism (e.g. supervisor process, platform service) is an implementation detail.
* **Structured outputs:** **Best-effort** when constrained decoding is unavailable; the **validate-and-retry** loop described later is an acceptable product bar. There is **no** requirement to guarantee schema-valid output on every response when the backend cannot enforce it.
* **Multi-model:** **Simultaneous** models; the gateway **routes by the request `model` field** (resolved from Hugging Face id, revision, or alias).
* **Platforms:** **Aim for Windows, macOS, and Linux**; resolve practical gaps as they appear (for example, if a backend is Linux-only, document it and decide per release).
* **Implementation language:** **Python** for the CLI and gateway for now; optional thin Rust or Go layers later.
* **Distribution:** Optimize for **user convenience**; the goal is that **installing `installm` leads to a usable stack** (exact bundling—extras, installers, optional runtime hooks—is TBD). Working name: **`installm`**.
* **Hardware:** **GPU support is required** for the intended full-quality path; **automatic backend selection** is explicitly **deferred** until implementation uncovers constraints.
* **Embeddings:** The same **`installm up --model`** flow as chat models. If a client calls **`/v1/embeddings`** with a model that is not embedding-capable, return **HTTP 400** with a clear message (no silent fallback to another model).
* **Model identifiers:** **`repo_id`**, optional **`@revision`** suffix, and **user-defined aliases** are in scope for MVP.
* **Compatibility target:** The **OpenAI Python SDK** is the **primary** reference client; meeting that bar is expected to cover **LangChain** and similar OpenAI-compatible stacks.
* **Ports and lifecycle:** Use **conventional, documented defaults** (for example a single gateway port such as **8000**, configurable). **`installm down` stops only processes that this installm instance started** (tracked in manifest or equivalent). **Logging:** defer detail; direction is **one combined log** with optional filters (for example time range) later.
* **Testing:** **CI** focuses on **contract tests** and paths that do not require a GPU. **GPU-heavy integration** is validated **manually** on a developer machine (for example RTX 4060).
* **API contract:** Follow the **endpoint set and shapes defined in this PRD** and the cited OpenAI / Open Responses references.
* **Hugging Face scope (MVP):** **No gated models** (no `HF_TOKEN` / license-gated download flows required for the first release); public-repo and standard-cache assumptions only.

---

## System architecture and core workflows

### Recommended high-level architecture

A practical, evolvable architecture is:

#### `installm` CLI (control plane)

* downloads models (HF cache / local directory)
* manages a local manifest of “installed models”
* starts/stops one or more inference backends
* starts the gateway and backends as a **supervised / daemon-appropriate** deployment (survives terminal close on supported setups)
* prints: Base URL, bind address, ready-to-copy environment variables, and “try it with curl” commands

#### `installm` gateway (data plane)

An HTTP server that exposes OpenAI-compatible endpoints and proxies/normalizes to backends. This is where you enforce consistent streaming format, tool-call shaping, and structured-output behavior.

#### Pluggable backends (you can support more than one)

* **Ollama backend**: excellent for GGUF/desktop use and already provides OpenAI-compatible endpoints, including `/v1/responses` (non-stateful flavor).
* **Transformers “serve” backend**: useful because it supports `/v1/chat/completions`, `/v1/responses`, `/v1/models`, and even a `/load_model` SSE endpoint for showing loading progress. But as of its docs, tool calling is limited to the Qwen family, so you cannot rely on it alone to satisfy “tool calling for any model.”
* **vLLM backend (strongly recommended for “must-have” features)**: provides an OpenAI-compatible server and supports structured outputs and tool calling mechanisms (including constrained decoding modes for tool calling).

### Key insight

Your non-negotiables *(structured outputs + tool calling across models)* push you toward using a backend with constrained decoding / structured output support (like vLLM), or implementing a tool/JSON enforcement layer yourself. vLLM already documents both.

### The “one command” workflow (end-to-end)

When the user runs:

```bash
installm up --model A --model B
```

the CLI should do:

1. **Resolve model IDs and revisions**

   * Accept `repo_id` plus optional revision pinning.

2. **Download model artifacts**

   * Use Hugging Face download primitives that support caching and incremental updates (so reruns are fast).

3. **Choose backend per model**

   * Example logic: if model is GGUF → Ollama; if GPU available and model is Transformers weights → vLLM; else fallback (CPU mode).

4. **Start services**

   * Start gateway first (so health is reachable), then start backends and register them.

5. **Warm-up optional**

   * If backend supports a “preload” endpoint, call it so first request isn’t slow. Transformers serve exposes `/load_model` with SSE progress events (useful UX inspiration).

6. **Print “how to use”**

   * Base URL examples are a known UX pattern in OpenAI-compatible setups.

---

## API spec and schemas

This section is written so you and Cursor can treat it as the “contract.” It defines what InstaLLM must accept/return.

### API compatibility philosophy (important)

* Be strict on output shape *(responses must match spec closely)*.
* Be liberal in input *(accept extra fields and ignore them when safe)*.
* Where the spec is large (e.g., Responses API), implement an explicit subset and return a clear `400` error for unsupported fields.
* **Primary client target:** validate behavior against the **OpenAI Python SDK** first; other stacks (e.g. LangChain) that speak the same OpenAI-compatible HTTP shapes should follow naturally.

### Endpoints to implement in MVP

| Endpoint                    | Purpose (plain English)                            | Required for MVP | Notes                                                 |
| --------------------------- | -------------------------------------------------- | ---------------: | ----------------------------------------------------- |
| `GET /health`               | “Is the server up?”                                |              Yes | Simple JSON.                                          |
| `GET /v1/models`            | “What models can I call?”                          |              Yes | Match OpenAI list models shape.                       |
| `POST /v1/chat/completions` | “Chat with the model”                              |              Yes | Primary compatibility endpoint.                       |
| `POST /v1/responses`        | Modern unified endpoint + semantic streaming       |              Yes | Follow Open Responses-style event streaming patterns. |
| `POST /v1/embeddings`       | Turn text into vectors (for semantic search / RAG) |              Yes | Match OpenAI schema.                                  |
| `POST /installm/load_model` | Warm-up / preload a model with progress            |     Nice-to-have | Copy Transformers serve idea (SSE progress).          |

### `GET /v1/models` schema

Return the OpenAI shape:

```json
{
  "object": "list",
  "data": [
    { "id": "model-id", "object": "model", "created": 1686935002, "owned_by": "organization-owner" }
  ]
}
```

This is explicitly what OpenAI documents for list models.

#### InstaLLM mapping recommendation

* `id`: canonical **Hugging Face repo id**, optionally with **revision suffix** (e.g. `org/model@rev`) when pinned. **User-defined aliases** map to one canonical id for requests and listing (list behavior: expose alias, canonical id, or both—pick one convention and document it in implementation).
* `created`: a timestamp when the model was added to your installm manifest (not necessarily HF publish time).
* `owned_by`: `installm` (or `local`).

### `POST /v1/embeddings` schema (MVP)

OpenAI defines:

* request: `model`, `input` (string or array), optional `dimensions`, optional `encoding_format`
* response: `object: "list"`, `data[]` with `{object:"embedding", embedding:[...], index:n}`, and `usage`.

#### InstaLLM constraints

Embedding support depends on having an embedding-capable backend/model. Models are brought up with the same **`installm up --model`** flow as chat models. If the resolved model is not embedding-capable (for example a chat-only checkpoint), return HTTP `400` with a clear error message (e.g. `"model does not support embeddings"`). Do **not** silently fall back to another model.

### `POST /v1/chat/completions` schema (MVP subset)

OpenAI’s request supports many fields (`messages`, tool calling, `response_format`, etc.). The minimum subset you must implement includes:

* `model: string`
* `messages: [{role, content, ...}]`
* `stream: boolean` *(if true → SSE stream of `chat.completion.chunk` objects)*
* **tool calling**:

  * `tools: [{type:"function", function:{name, description, parameters, strict?}}]`
  * `tool_choice: "auto" | "none" | "required" | {type:"function", function:{name}}`
* **structured outputs**:

  * `response_format: {type:"json_object"} | {type:"json_schema", json_schema:{name, description?, schema, strict?}}`

For streaming responses, OpenAI documents chunk objects where deltas include content and/or `tool_calls`; tool calls can arrive incrementally, so clients must accumulate argument fragments.

### `POST /v1/responses` schema (MVP subset)

Because the official Responses API schema is huge, InstaLLM should implement a deliberate subset focused on your needs.

#### Accept

* `model: string`
* `input: string | array of input items`
* `instructions: string` *(maps to a system/developer message)*
* `tools: []` *(function tools)*
* `tool_choice: "auto" | "none" | "required" | {name: string}` *(you can normalize to the official shape internally)*
* `stream: boolean`
* `max_output_tokens`, `temperature`, `top_p`

This matches what multiple OpenAI-compatible servers treat as supported, including Ollama’s compatibility docs.

#### Return

A response object with an `output` array containing items like:

* `message` items with `content[]` parts containing `output_text`
* `function_call` items when tool calling occurs *(with arguments streamed as deltas)*

### Streaming format

For `stream=true`, use Server-Sent Events (`Content-Type: text/event-stream`). The Open Responses spec defines semantic events like:

* `response.created`
* `response.output_item.added`
* `response.output_text.delta`

…and states that each SSE event includes JSON on a `data:` line.

`response.js` provides a concrete implementation pattern:

* it runs everything as a stream and increments `sequence_number`
* for streaming requests it writes `data: <json>\n\n` events
* for non-stream requests it collects until `response.completed` and returns final JSON

#### InstaLLM recommendation (compat-safe)

Emit both:

```text
event: <type>
data: <json>
```

…and then terminate the stream by both:

* emitting `response.completed`, and
* emitting `data: [DONE]` just before closing the connection *(some clients expect this pattern)*.

The Open Responses spec explicitly calls for `[DONE]` as the terminal literal.

---

## Implementation details for the hard parts

This is the “how to build it” portion, written in a way Cursor can directly turn into tasks.

### Streaming (SSE) implementation

**Plain-language goal:** Instead of waiting for the full model output, the server sends many small messages (“chunks” or “events”), so the user sees text appear gradually.

#### Two streaming “dialects” you must support

* **Chat Completions streaming**: stream of `chat.completion.chunk` objects; each chunk includes `choices[].delta` and can include `delta.tool_calls`.
* **Responses streaming**: semantic events (`response.created`, `response.output_text.delta`, …) with `sequence_number`.

#### Implementation pattern (borrow from `response.js`)

Implement your internal generation pipeline as an async generator of events. Then:

* if `stream=true`, write each event as SSE
* if `stream=false`, consume the generator until “completed”, then return a single JSON response

This keeps one codepath and reduces bugs *(exactly why `response.js` does it)*.

#### FastAPI note

If you use FastAPI, FastAPI has an official SSE tutorial and uses the standard `text/event-stream` format.

### Tool calling implementation

**Plain-language goal:** In some requests, the model shouldn’t answer directly; it should say “call this function,” with arguments. Your app (or LangChain) runs the function and sends results back.

#### Key protocol behaviors you must implement

1. **Request accepts tools**

   * a list of functions defined by JSON Schema.
   * This is the same structure used by OpenAI, Ollama, HF, and vLLM.

2. **Response includes tool calls**

   * **Chat Completions**: message contains `tool_calls` with `id`, `type:"function"`, and `function:{name, arguments}`. In streaming, these can arrive in pieces.
   * **Responses**: output items include `function_call` plus streaming events like `response.function_call_arguments.delta/done`.
   * `response.js` shows how to translate chat `tool_call` deltas into Responses events.

3. **Tool execution is external**

   * your runtime executes it (or LangChain does).
   * Hugging Face and LangChain both clearly state this runtime responsibility.

#### How to support “any model” realistically

Here is the hard truth you should put into the PRD as an explicit constraint:

> “Tool calling support” in the API sense is always possible (accept/emit `tool_call` objects).
> But whether the model chooses the right tool or produces good arguments depends on the model and/or constrained decoding. OpenAI’s own docs warn tool arguments may be invalid; you must validate.

Additional realities:

* Some servers explicitly limit tool calling to certain model families (Transformers serve says tool calling is currently limited to Qwen family).

So InstaLLM should define three **tool calling modes**:

1. **Native mode**

   * backend natively supports tool calls for that model; passthrough.

2. **Constrained mode** *(recommended default where available)*

   * force a valid tool-call structure using constrained decoding / structured outputs mechanisms.
   * vLLM documents that for `tool_choice="required"` and “named function calling,” it uses structured-output backends to guarantee a validly-parsable tool call.

3. **Prompt-and-parse fallback**

   * if backend cannot constrain tool output, you can still ask the model to output tool call JSON and parse it.
   * Hugging Face Transformers includes a “response parsing” system based on response schemas and regex + JSON parsing that can extract tool calls from raw text.

#### Critical safety/performance note

From HF ecosystem reality: parsing tool calls with complex regex can hang on malformed/truncated outputs *(catastrophic backtracking)*. This is a real issue reported against Qwen schemas when outputs are truncated. Your PRD should include mitigations:

* timeouts
* max-length caps
* safe regex patterns

### Structured outputs implementation

**Plain-language goal:** The caller wants the model’s final answer as JSON (not prose), often matching a schema so code can reliably read it.

OpenAI defines “Structured Outputs” as schema-constrained JSON generation via:

```json
response_format: {"type":"json_schema", "json_schema": {...}}
```

…and also supports “JSON mode” via:

```json
response_format: {"type":"json_object"}
```

#### Backend-driven enforcement (recommended)

If you choose a backend that supports constrained decoding for JSON schema, you can actually enforce schema compliance:

* vLLM documents JSON schema structured outputs and shows examples using `response_format` with a JSON schema (often derived from a Pydantic model).

#### InstaLLM PRD requirement

Implement:

* `response_format.type="json_object"`: **guarantee valid JSON** when the backend can enforce it; otherwise **best-effort** via instruction + parse + **validate-and-retry** as below.
* `response_format.type="json_schema"` with `strict`: **guarantee** JSON matches the schema **when the backend supports constrained decoding**; otherwise use **best-effort** validate-and-retry (no hard guarantee on every response).

#### Fallback strategy (when no constrained decoding is available)

Implement a “validate and retry” loop:

1. Inject a strong instruction: “Output ONLY valid JSON matching this schema… no markdown.”
2. Generate.
3. Attempt to parse JSON.
4. Validate against schema (`jsonschema` library).
5. If invalid, retry `N` times with the validation error message included *(so the model can fix)*.

This is not as bulletproof as constrained decoding, but it provides “practical” structured output for models without special decoding support, and is widely used.

### Downloading and managing models

#### Download primitive

Use Hugging Face Hub download functionality with caching:

* `snapshot_download` supports patterns (`allow`/`ignore`) and supports both cache-based downloads and `local_dir` sync behavior that avoids re-downloading unchanged files.

#### State management

Store a local manifest, e.g.:

```text
~/.installm/state.json
```

This should track:

* installed models
* revisions
* backend selection
* ports
* last used
* capabilities

#### Model discovery

Expose the manifest via `/v1/models` in OpenAI compatible list-shape.

### Multi-model serving

You have two viable designs:

#### Design A (simpler UX, fewer processes)

Single backend that supports on-demand multi-model loading.

Transformers serve does this:

* loads models on demand
* unloads after timeout
* `/v1/models` endpoint scans local HF cache
* also exposes `/load_model` SSE progress

But tool calling is limited there, so you’d still need your gateway layer for “any model” tool calling.

#### Design B (more universal features)

One backend process per model + gateway router.

This maps cleanly to vLLM (and Ollama) and makes capability differences explicit. LangChain itself documents vLLM can be run as an OpenAI-like server, and that model features depend on the hosted model.

InstaLLM’s gateway would route requests based on the `model` field.

For your “production-ready” framing, Design B is usually easier to evolve toward Kubernetes/cluster deployment later.

**MVP commitment:** multi-model means **several models can be up at once** and the gateway **routes by `model` name** (canonical HF id, revision suffix, or alias). Single-model-only or “switch active model” is **not** the MVP bar.

---

## Phased implementation plan with testable milestones

Each phase below can be built and tested independently. “MVP weeks” are rough timelines for a single developer with decent Python familiarity.

### Phase foundation

#### Deliverables

* Repo scaffold, CLI skeleton, configuration file format, local state manifest.
* `installm --help` shows commands and examples.
* Configuration for **listen/bind address** (local vs `0.0.0.0`) and **daemon-oriented** `up` / `down` (see baseline decisions).

#### CLI commands (minimum)

```bash
installm up --model <id> [--model <id>...]
installm ls
installm down
installm logs
```

#### Tests

* Unit test: manifest read/write.
* Smoke test: `installm ls` works on fresh machine.
* CI does **not** assume a GPU; GPU scenarios are **manual** until otherwise specified.

**Estimate:** 1 week.

### Phase download and model inventory

#### Deliverables

* Download models using HF caching.
* Implement `installm pull <model>` and integrate into `up`.
* Use `snapshot_download` to fetch repos and rely on HF’s cache/local sync behavior.

#### Tests

* `installm pull gpt2` (or any small model) completes, and rerun shows near-zero download.
* Manifest records model and optional revision.

**Estimate:** 1 week.

### Phase OpenAI-compatible gateway (non-streaming)

#### Deliverables

HTTP server that implements:

* `GET /health`
* `GET /v1/models` (OpenAI list shape)
* `POST /v1/chat/completions` with `stream=false` (basic text)

A single “dummy backend” that returns canned responses (for contract tests).

#### Tests

* Contract tests validating response JSON shapes against your own JSON schemas.
* Curl smoke tests.

**Estimate:** 1–2 weeks.

### Phase streaming for Chat Completions

#### Deliverables

* `/v1/chat/completions` supports `stream=true`, emitting SSE chunks in the documented schema, including partial delta objects.

#### Hard part to implement correctly

Tool calls in streaming arrive in fragments; you must accumulate them *(at least in your backend adapter and in your tests)*. vLLM’s own example shows a correct approach: store tool calls by id and append `function.arguments` chunks.

#### Tests

* Snapshot test of SSE stream: verify at least:

  * first chunk includes `role:"assistant"` or appropriate delta
  * later chunks include `delta.content`
  * stream ends correctly
* “Slow client” test: ensure you flush properly and don’t buffer everything.

**Estimate:** 1–2 weeks.

### Phase tool calling protocol support

#### Deliverables

* Accept `tools` and `tool_choice` in `/v1/chat/completions`.
* Emit `tool_calls` in response messages when backend returns them.
* Implement “tool-result messages”: accept prior tool calls and tool outputs in messages (`role tool`), consistent with standard tool loops.

#### Backend strategy

If using vLLM, implement:

* passthrough for auto tool mode
* support `tool_choice="required"` and named function mode for guaranteed parseable outputs

If using prompt-and-parse fallback, implement Transformers `parse_response` support when possible and include timeouts to avoid regex hangs.

#### Tests

* Integration test with a known tool-capable model *(choose one model and lock it as the CI test model)*.
* Validate that tool calls appear in the response object in the correct shape.

**Estimate:** 2–3 weeks.

### Phase structured outputs

#### Deliverables

Support `response_format` in Chat Completions:

* `json_object`
* `json_schema` with schema payload

Prefer backend-enforced schema compliance when available *(vLLM supports JSON schema structured outputs)*.

Implement fallback validate-and-retry loop otherwise.

#### Tests

* JSON schema compliance test: force a schema and verify the final `message.content` parses and validates.
* Failure-mode test: invalid schema returns `400`.

**Estimate:** 2–3 weeks.

### Phase Responses API support

#### Deliverables

* Implement `/v1/responses` request/response subset, including semantic streaming events with `sequence_number`.
* Build it using the `response.js` “always stream internally” pattern.

#### Tests

* Streaming test verifies correct event ordering:

  * `response.created`
  * `response.in_progress`
  * content/tool events
  * `response.completed`
* Non-stream mode returns final response object.

**Estimate:** 2–4 weeks.

### Phase optional auth and metrics

#### Auth

Optional `--api-key` that requires:

```http
Authorization: Bearer <key>
```

for all `/v1/*` endpoints *(except `/health`)*.

This is compatible with Open Responses spec expectation that `Authorization` is present.

#### Metrics

Optional `/metrics` (Prometheus) and request logging.

**Estimate:** 1–2 weeks.

---

## Final recommendation and what to build first

### Recommendation

Base InstaLLM on a **gateway + pluggable runtimes** approach, and prioritize a **vLLM-based runtime** for “must-have” features *(tool calling + structured outputs)*, while treating **Ollama** and **Transformers serve** as additional runtime adapters.

### Why

* Tool calling + structured outputs are your non-negotiables, and vLLM explicitly supports structured outputs via JSON schema and provides tool calling modes that can guarantee valid tool-call outputs in constrained modes.
* Tool calling support is not reliably universal in other stacks (Transformers serve even documents a current model-family limitation).
* Ollama is a great complementary adapter and already implements a subset of OpenAI endpoints including `/v1/responses`, plus the well-known “just point `base_url` to localhost” usage.

### Mermaid diagram of the recommended integration

```mermaid
flowchart LR
  U[User] -->|installm up --model ...| CLI[installm CLI]
  CLI -->|snapshot_download / hf cache| HF[(Local HF cache)]
  CLI --> GW[installm Gateway : OpenAI-compatible API]
  CLI -->|start runtime per model| R1[vLLM runtime (model A)]
  CLI -->|start runtime per model| R2[Ollama runtime (model B)]
  GW -->|route by request.model| R1
  GW -->|route by request.model| R2
  App[LangChain / OpenAI SDK client] -->|base_url=http://host:port/v1| GW
  GW -->|SSE stream| App
  App -->|tool execution in app| Tools[(Your tools)]
  App -->|tool results as tool messages| GW
```

This diagram encodes the main “responsibility split”: InstaLLM serves the API and formats tool calls; the client/app executes tools and sends results back.

---

## Primary/official sources referenced (copyable)

### Hugging Face Hub downloads (snapshot_download / caching)

* [https://huggingface.co/docs/huggingface_hub/en/guides/download](https://huggingface.co/docs/huggingface_hub/en/guides/download)

### Transformers Serve CLI (OpenAI-compatible endpoints, /load_model SSE, tool calling note)

* [https://huggingface.co/docs/transformers/main/serving](https://huggingface.co/docs/transformers/main/serving)
* [https://github.com/huggingface/transformers/blob/main/docs/source/en/serve-cli/serving.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/serve-cli/serving.md)

### Transformers tool use + the “model cannot execute tools” statement

* [https://huggingface.co/docs/transformers/chat_extras](https://huggingface.co/docs/transformers/chat_extras)

### Transformers response parsing (parse_response, response_schema)

* [https://huggingface.co/docs/transformers/en/chat_response_parsing](https://huggingface.co/docs/transformers/en/chat_response_parsing)

### response.js (Responses API streaming/event approach)

* [https://github.com/huggingface/responses.js](https://github.com/huggingface/responses.js)
* [https://raw.githubusercontent.com/huggingface/responses.js/main/src/routes/responses.ts](https://raw.githubusercontent.com/huggingface/responses.js/main/src/routes/responses.ts)

### Ollama OpenAI compatibility and supported endpoints

* [https://ollama.com/blog/openai-compatibility](https://ollama.com/blog/openai-compatibility)
* [https://docs.ollama.com/api/openai-compatibility](https://docs.ollama.com/api/openai-compatibility)

### Ollama tool support

* [https://ollama.com/blog/tool-support](https://ollama.com/blog/tool-support)

### OpenAI API reference (models, embeddings, chat completions, streaming chunks, structured outputs, streaming guide)

* [https://developers.openai.com/api/reference/resources/models/methods/list/](https://developers.openai.com/api/reference/resources/models/methods/list/)
* [https://developers.openai.com/api/reference/resources/embeddings/methods/create/](https://developers.openai.com/api/reference/resources/embeddings/methods/create/)
* [https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/)
* [https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events/](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events/)
* [https://developers.openai.com/api/docs/guides/structured-outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
* [https://developers.openai.com/api/docs/guides/streaming-responses](https://developers.openai.com/api/docs/guides/streaming-responses)

### Open Responses streaming/events specification

* [https://www.openresponses.org/specification](https://www.openresponses.org/specification)

### LangChain docs about using OpenAI-compatible servers (base URL patterns, vLLM integration, tool call execution)

* [https://docs.langchain.com/langsmith/custom-openai-compliant-model](https://docs.langchain.com/langsmith/custom-openai-compliant-model)
* [https://docs.langchain.com/oss/python/integrations/chat/vllm](https://docs.langchain.com/oss/python/integrations/chat/vllm)
* [https://docs.langchain.com/oss/python/langchain/frontend/tool-calling](https://docs.langchain.com/oss/python/langchain/frontend/tool-calling)

### vLLM OpenAI-compatible server + tool calling + structured outputs

* [https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html](https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html)
* [https://docs.vllm.ai/en/latest/features/tool_calling/](https://docs.vllm.ai/en/latest/features/tool_calling/)
* [https://docs.vllm.ai/en/latest/features/structured_outputs/](https://docs.vllm.ai/en/latest/features/structured_outputs/)
