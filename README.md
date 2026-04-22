# InstaLLM

[![PyPI version](https://badge.fury.io/py/installm.svg)](https://pypi.org/project/installm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**One-command deployment of OpenAI-compatible APIs for open-source LLMs.**

InstaLLM is a developer tool that turns any open-source large language model into a production-ready API server with a single CLI command. It is designed for developers building AI applications who want the flexibility of open-source models with the convenience of the OpenAI API contract.

```bash
pip install installm
installm up --model Qwen/Qwen2.5-7B-Instruct
```

Your API is now live at `http://localhost:8000`. Point any OpenAI SDK or LangChain app at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## Features

| Feature | Description |
|:---|:---|
| **One-command deployment** | `installm up --model <model>` — that's it |
| **OpenAI-compatible API** | Drop-in replacement: change `base_url`, keep all your code |
| **Four backends** | vLLM (GPU), Transformers (CPU/MPS/CUDA), llama.cpp (GGUF), Ollama |
| **Auto backend selection** | Picks the best backend for your hardware automatically |
| **SSE Streaming** | Real-time token streaming via Server-Sent Events |
| **Tool Calling** | Native for vLLM/Ollama; prompt-and-parse fallback for Transformers/llama.cpp |
| **Structured Outputs** | `json_object` and `json_schema` with validate-and-retry fallback |
| **Responses API** | Semantic streaming events following the Open Responses spec |
| **Multi-model** | Deploy multiple models simultaneously, gateway routes by `model` field |
| **Model Aliases** | Map short names to long model IDs for convenience |
| **API Key Authentication** | Optional Bearer token auth, OpenAI-compatible |
| **Docker support** | CPU and GPU Dockerfiles included |

---

## Installation

```bash
# Base install (Ollama backend only, Ollama must be installed separately)
pip install installm

# With Transformers backend (CPU / MPS / CUDA)
pip install "installm[transformers]"

# With vLLM backend (Linux + NVIDIA GPU only)
pip install "installm[vllm]"

# With llama.cpp backend (GGUF models)
pip install "installm[llamacpp]"

# Everything
pip install "installm[transformers,vllm,llamacpp]"
```

**Requirements:** Python 3.10+

---

## Quick Start

### 1. Start a model

```bash
# Auto-selects the best backend for your hardware
installm up --model Qwen/Qwen2.5-7B-Instruct

# Force a specific backend
installm up --model Qwen/Qwen2.5-7B-Instruct --backend transformers

# Custom host and port
installm up --model Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8080
```

### 2. Use it — no code changes needed

```python
from openai import OpenAI

# Just change base_url — everything else stays the same
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain transformers in one paragraph."}],
)
print(response.choices[0].message.content)
```

### 3. Streaming

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Write a haiku about open-source AI."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### 4. Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Hong Kong?"}],
    tools=tools,
    tool_choice="auto",
)
tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, tool_call.function.arguments)
```

### 5. Structured Outputs

```python
import json

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Give me a person with name and age"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        },
    },
)
person = json.loads(response.choices[0].message.content)
print(person)  # {"name": "Alice", "age": 30}
```

### 6. Model Aliases

```bash
# Create a short alias for a long model ID
installm alias qwen Qwen/Qwen2.5-7B-Instruct

# Now use the alias in API calls
curl http://localhost:8000/v1/chat/completions \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Hi"}]}'

# Remove an alias
installm unalias qwen
```

### 7. Multi-model Serving

```bash
installm up --model Qwen/Qwen2.5-7B-Instruct --model mistralai/Mistral-7B-Instruct-v0.3
```

Both models are accessible through the same API — the gateway routes requests based on the `model` field.

### 8. Authentication

InstaLLM supports optional API key authentication that mirrors the OpenAI API pattern:

```bash
# Generate a key
installm auth create --label "dev-laptop"
# >> Key: sk-installm-a1b2c3d4...  (save this!)

# Start the server with auth enabled
installm up --model Qwen/Qwen2.5-7B-Instruct --require-auth

# Or enable via environment variable
export INSTALLM_REQUIRE_AUTH=1
installm up --model Qwen/Qwen2.5-7B-Instruct
```

Clients authenticate exactly like they do with OpenAI — the `api_key` parameter just works:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-installm-a1b2c3d4...",  # Your generated key
)
```

Key management:

```bash
installm auth ls        # List active keys (prefix only)
installm auth revoke <id>  # Revoke a key
```

When auth is not enabled (the default), all requests pass through without any key — fully backward compatible.

### 9. Framework Compatibility

InstaLLM works with any framework that supports the OpenAI API:

**LangChain:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="Qwen/Qwen2.5-7B-Instruct",
)
print(llm.invoke("What is InstaLLM?").content)
```

**CrewAI:**
```python
from crewai import LLM

llm = LLM(
    model="openai/Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)
```

---

## CLI Reference

| Command | Description |
|:---|:---|
| `installm up --model <id> [--model <id>...]` | Pull model(s) and start the API server |
| `installm pull --model <id>` | Download a model without starting the server |
| `installm ls` | List all downloaded models and aliases |
| `installm down` | Stop the running server |
| `installm logs` | Show recent server logs |
| `installm alias <name> <model_id>` | Create a short alias for a model ID |
| `installm unalias <name>` | Remove a model alias |
| `installm auth create [--label]` | Generate a new API key |
| `installm auth ls` | List active API keys (prefix only) |
| `installm auth revoke <id>` | Revoke an API key |

### `installm up` options

| Option | Default | Description |
|:---|:---|:---|
| `--model`, `-m` | (required) | HuggingFace model ID (repeatable for multi-model) |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port number |
| `--backend` | auto | Force a backend: `transformers`, `vllm`, `ollama`, or `llamacpp` |
| `--require-auth` | off | Require API key authentication for all requests |

---

## API Reference

| Endpoint | Method | Description |
|:---|:---|:---|
| `/health` | GET | Liveness check |
| `/v1/models` | GET | List loaded models and aliases (OpenAI format) |
| `/v1/chat/completions` | POST | Chat completion (streaming and non-streaming) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/responses` | POST | Responses API with semantic streaming events |

---

## Backends

InstaLLM auto-selects the best available backend in this order:

1. **vLLM** — highest throughput, requires Linux + NVIDIA GPU with CUDA
2. **Transformers** — universal, works on CPU / Apple MPS / CUDA
3. **llama.cpp** — efficient GGUF inference, works on CPU and GPU
4. **Ollama** — requires Ollama to be installed separately

You can force a specific backend with `--backend`:

```bash
installm up --model Qwen/Qwen2.5-7B-Instruct --backend transformers
```

### Backend capabilities

| Feature | vLLM | Transformers | llama.cpp | Ollama |
|:---|:---:|:---:|:---:|:---:|
| Tool calling | Native | Gateway | Gateway | Native |
| Structured outputs | Native | Gateway | Gateway | Native |
| Streaming | Yes | Yes | Yes | Yes |
| Embeddings | Yes | Yes | Yes | Yes |
| GPU required | Yes | No | No | No |
| Platform | Linux | All | All | All |

> **Native** means the inference engine enforces the constraint at the model level. **Gateway** means InstaLLM handles it transparently — tool calls are injected via a system prompt and parsed from the model output; structured outputs are validated against the schema with automatic retries. From the API caller's perspective, both modes are identical.

### Platform notes

- **vLLM** raises a clear error on Windows/macOS or when no CUDA GPU is detected
- **Transformers** auto-detects CUDA > MPS > CPU
- **llama.cpp** works with GGUF model files; auto-resolves from HuggingFace cache
- **Ollama** requires the Ollama daemon to be running (`ollama serve`)

---

## Docker

### CPU

```bash
docker build -t installm .
docker run -p 8000:8000 installm up --model Qwen/Qwen2.5-0.5B-Instruct
```

### GPU (NVIDIA)

```bash
docker build --build-arg BASE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 -t installm-gpu .
docker run --gpus all -p 8000:8000 installm-gpu up --model Qwen/Qwen2.5-7B-Instruct
```

### Docker Compose

```bash
docker compose up
```

---

## Project Structure

```
src/installm/
├── __init__.py          # Version
├── auth.py              # API key generation, hashing, validation
├── cli.py               # Click CLI (up, down, ls, pull, alias, auth, logs)
├── config.py            # State manifest, aliases (~/.installm/state.json)
├── download.py          # HuggingFace Hub integration
├── backends/
│   ├── __init__.py      # Backend registry and auto-selection
│   ├── base.py          # Abstract base class
│   ├── transformers.py  # HF Transformers backend
│   ├── vllm.py          # vLLM backend
│   ├── llamacpp.py      # llama.cpp backend (GGUF)
│   └── ollama.py        # Ollama backend
└── gateway/
    ├── __init__.py
    ├── app.py           # FastAPI app, backend registry, server launcher
    ├── middleware.py     # Auth middleware (Bearer token validation)
    ├── schemas.py       # Pydantic models (OpenAI contract)
    ├── streaming.py     # SSE helpers
    ├── tools.py         # Tool calling prompt injection and parsing
    ├── structured.py    # JSON mode and validate-and-retry
    └── routes/
        ├── __init__.py
        ├── models.py    # GET /v1/models
        ├── chat.py      # POST /v1/chat/completions
        ├── embeddings.py# POST /v1/embeddings
        └── responses.py # POST /v1/responses
```

---

## Testing

```bash
# Install test dependencies
pip install "installm[transformers]" pytest pytest-asyncio httpx

# Run unit tests (fast, no model download, no GPU needed)
pytest tests/ --ignore=tests/test_integration_live.py --ignore=tests/test_e2e_qwen.py -v

# Run live integration tests (downloads a 2.5MB test model)
pytest tests/test_integration_live.py -v

# Run full e2e tests with OpenAI SDK + LangChain (downloads Qwen2.5-0.5B)
pytest tests/test_e2e_qwen.py -v

# Run everything
pytest tests/ -v
```

### Test coverage

| Test file | What it covers |
|:---|:---|
| `test_config.py` | State manifest CRUD, server info lifecycle |
| `test_alias.py` | Alias set/remove/resolve, backward compat |
| `test_cli.py` | CLI help, ls, pull commands |
| `test_download.py` | HF Hub download, caching |
| `test_backends/test_ollama.py` | Ollama backend (mocked) |
| `test_backends/test_transformers.py` | Transformers backend (mocked) |
| `test_backends/test_vllm.py` | vLLM backend (mocked) |
| `test_backends/test_llamacpp.py` | llama.cpp backend (mocked) |
| `test_gateway/test_health.py` | Health endpoint |
| `test_gateway/test_models.py` | Models list endpoint |
| `test_gateway/test_chat.py` | Chat completions (non-streaming, streaming, tools, structured) |
| `test_gateway/test_embeddings.py` | Embeddings endpoint |
| `test_gateway/test_responses.py` | Responses API (non-streaming, streaming events) |
| `test_gateway/test_tools_and_structured.py` | Tool prompt builder, JSON parser, validate-and-retry |
| `test_auth.py` | Key CRUD, validation, middleware (401/200), CLI commands |
| `test_integration_live.py` | Live test with tiny-gpt2 model |
| `test_e2e_qwen.py` | Full e2e with OpenAI SDK, LangChain, tool calling, JSON mode |

---

## Future Work

- **Per-key rate limiting** — throttle requests per API key
- **Prometheus Metrics** — `/metrics` endpoint for monitoring
- **Model Routing** — route requests to different models based on rules
- **Observability Dashboard** — request logs, latency metrics, token usage
- **TensorRT-LLM backend** — NVIDIA-optimised inference for production

---

## License

MIT
