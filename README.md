# InstaLLM

**One command. Any open-source model. OpenAI-compatible API.**

InstaLLM is a developer tool that deploys any open-source large language model as a production-ready, OpenAI-compatible REST API. If you are building an AI application and want to use open-source models instead of (or alongside) commercial APIs, InstaLLM removes all the infrastructure boilerplate so you can focus on your application.

```bash
# Install
pip install installm

# Pull a model and start the API server
installm up --model meta-llama/Llama-3.1-8B-Instruct

# Your app now talks to http://localhost:8000 — no code changes needed
```

---

## Why InstaLLM?

Building AI applications with open-source models is powerful but painful. You need to:

1. Choose and configure an inference engine (vLLM, Transformers, Ollama...)
2. Write a server that exposes the model as an HTTP API
3. Implement streaming, tool calling, and structured outputs yourself
4. Make sure all of this works with your AI framework (LangChain, CrewAI, OpenAI SDK...)

InstaLLM handles all of this with a single command. It auto-selects the best backend for your hardware and exposes a fully OpenAI-compatible API, so any code written against the OpenAI SDK works unchanged.

---

## Features

| Feature | Description |
|:---|:---|
| **One-command deployment** | `installm up --model <model>` — that's it |
| **OpenAI-compatible API** | Drop-in replacement: change `base_url`, keep all your code |
| **Auto backend selection** | vLLM on Linux+GPU, Transformers on CPU/MPS, Ollama as fallback |
| **SSE Streaming** | Real-time token streaming via Server-Sent Events |
| **Tool Calling** | Native for capable backends; prompt-and-parse fallback for others |
| **Structured Outputs** | `json_object` and `json_schema` with validate-and-retry loop |
| **Responses API** | OpenAI Responses API with semantic streaming events |
| **Multi-model** | Run multiple models simultaneously on different ports |
| **Docker support** | Single container deployment, GPU-ready |

---

## Installation

```bash
# Core install (Ollama backend only)
pip install installm

# With Transformers backend (CPU/MPS/CUDA)
pip install "installm[transformers]"

# With vLLM backend (Linux + NVIDIA GPU only)
pip install "installm[vllm]"

# Everything
pip install "installm[transformers,vllm]"
```

**Requirements:** Python 3.11+

---

## Quick Start

### 1. Start a model

```bash
# Auto-selects the best backend for your hardware
installm up --model meta-llama/Llama-3.1-8B-Instruct

# Force a specific backend
installm up --model meta-llama/Llama-3.1-8B-Instruct --backend transformers

# Custom host and port
installm up --model mistralai/Mistral-7B-Instruct-v0.3 --host 0.0.0.0 --port 8080
```

### 2. Use it — no code changes needed

```python
from openai import OpenAI

# Just change base_url — everything else stays the same
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain transformers in one paragraph."}],
)
print(response.choices[0].message.content)
```

### 3. Streaming

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
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
    model="meta-llama/Llama-3.1-8B-Instruct",
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
    model="meta-llama/Llama-3.1-8B-Instruct",
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

### 6. Framework Compatibility

InstaLLM works with any framework that supports the OpenAI API:

**LangChain:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="meta-llama/Llama-3.1-8B-Instruct",
)
print(llm.invoke("What is InstaLLM?").content)
```

**CrewAI:**
```python
from crewai import LLM

llm = LLM(
    model="openai/meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)
```

---

## CLI Reference

```
installm up       Start the API server for a model
installm down     Stop the running server
installm ls       List loaded models and server status
installm pull     Download a model from HuggingFace Hub
installm logs     Tail the server log
```

### `installm up`

```
Options:
  --model TEXT      HuggingFace model ID or local path  [required]
  --backend TEXT    Backend: auto | transformers | vllm | ollama  [default: auto]
  --host TEXT       Bind host  [default: 127.0.0.1]
  --port INTEGER    Bind port  [default: 8000]
  --help            Show this message and exit.
```

### `installm pull`

```
Options:
  --model TEXT      HuggingFace model ID to download  [required]
  --revision TEXT   Specific revision/branch  [default: main]
  --help            Show this message and exit.
```

---

## API Reference

InstaLLM exposes a fully OpenAI-compatible REST API. All endpoints accept and return JSON.

### `GET /health`

```json
{"status": "ok", "models_loaded": 1}
```

### `GET /v1/models`

Returns all loaded models in OpenAI format.

### `POST /v1/chat/completions`

Full OpenAI Chat Completions API. Supports:
- `stream: true` — Server-Sent Events streaming
- `tools` + `tool_choice` — function/tool calling
- `response_format` — `json_object` or `json_schema`

### `POST /v1/embeddings`

Generate text embeddings (requires an embedding-capable model).

### `POST /v1/responses`

OpenAI Responses API with semantic streaming events:
`response.created` → `response.in_progress` → `response.output_item.added` → `response.output_text.delta` (×N) → `response.output_text.done` → `response.output_item.done` → `response.completed`

---

## Backends

InstaLLM auto-selects the best backend based on your hardware. You can also specify one explicitly with `--backend`.

| Backend | Platform | Hardware | Best For |
|:---|:---|:---|:---|
| `vllm` | Linux only | NVIDIA GPU | Production, high throughput |
| `transformers` | Any | CPU / MPS / CUDA | Development, any hardware |
| `ollama` | Any | CPU / GPU | If Ollama is already installed |

**Auto-selection order:** vLLM (Linux + CUDA) → Transformers → Ollama

### Platform Notes

- **Windows:** vLLM is not supported. InstaLLM will automatically fall back to the Transformers backend and notify you.
- **macOS (Apple Silicon):** Transformers backend uses MPS acceleration automatically.
- **Linux + NVIDIA GPU:** vLLM is used by default for maximum throughput.

---

## Docker

```bash
# Build and run (CPU)
docker build -t installm .
docker run -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  installm up --model sshleifer/tiny-gpt2 --host 0.0.0.0

# With docker-compose
MODEL=meta-llama/Llama-3.1-8B-Instruct docker compose up

# GPU variant
MODEL=meta-llama/Llama-3.1-8B-Instruct docker compose --profile gpu up
```

---

## Project Structure

```
installm/
├── src/installm/
│   ├── cli.py              # Click CLI entry point
│   ├── config.py           # State/manifest management (~/.installm/)
│   ├── download.py         # HuggingFace Hub model download
│   ├── backends/
│   │   ├── base.py         # Backend abstract base class
│   │   ├── __init__.py     # Auto-selection logic
│   │   ├── transformers.py # HF Transformers backend
│   │   ├── vllm.py         # vLLM backend (Linux + NVIDIA)
│   │   └── ollama.py       # Ollama backend
│   └── gateway/
│       ├── app.py          # FastAPI app + server launcher
│       ├── schemas.py      # Pydantic request/response schemas
│       ├── streaming.py    # SSE helpers
│       ├── tools.py        # Tool calling: prompt injection + parsing
│       ├── structured.py   # Structured output: JSON enforcement + retry
│       └── routes/
│           ├── models.py   # GET /v1/models
│           ├── chat.py     # POST /v1/chat/completions
│           ├── embeddings.py # POST /v1/embeddings
│           └── responses.py  # POST /v1/responses
└── tests/
    ├── test_config.py
    ├── test_cli.py
    ├── test_download.py
    ├── test_backends/
    │   ├── test_ollama.py
    │   ├── test_transformers.py
    │   └── test_vllm.py
    ├── test_gateway/
    │   ├── test_health.py
    │   ├── test_models.py
    │   ├── test_chat.py
    │   ├── test_embeddings.py
    │   ├── test_responses.py
    │   └── test_tools_and_structured.py
    └── test_integration_live.py   # Real model tests (requires torch + transformers)
```

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all unit tests (no model download required)
pytest tests/ --ignore=tests/test_integration_live.py

# Run live integration tests (downloads ~5MB model on first run)
pytest tests/test_integration_live.py -v -s

# Run everything
pytest tests/
```

---

## Future Work

- **API Key Authentication** — per-key rate limiting and access control
- **Model Routing** — route requests to different models based on rules
- **Observability Dashboard** — request logs, latency metrics, token usage
- **llama.cpp backend** — ultra-low memory inference via GGUF models
- **TensorRT-LLM backend** — NVIDIA-optimised inference for production

---

## License

MIT
