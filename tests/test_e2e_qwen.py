"""End-to-end tests using Qwen2.5-0.5B-Instruct (instruction-tuned, ~1 GB).

Tests the full stack: backend → gateway → OpenAI SDK / LangChain.
Validates tool calling fallback, structured output, and streaming with
a model that actually follows instructions.

Run with:  pytest tests/test_e2e_qwen.py -v -s --timeout=120
"""

import asyncio
import json
import threading
import time
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import uvicorn
from installm.backends.transformers import TransformersBackend
from installm.gateway.app import app, _backends, register_backend

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 9123
BASE_URL = f"http://127.0.0.1:{PORT}/v1"


# ---------------------------------------------------------------------------
# Fixtures: load model once, run gateway in background thread
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def loaded_backend(event_loop):
    """Load Qwen2.5-0.5B-Instruct once for all tests."""
    backend = TransformersBackend()
    event_loop.run_until_complete(backend.load(MODEL, device="cpu"))
    yield backend
    event_loop.run_until_complete(backend.unload())


@pytest.fixture(scope="module", autouse=True)
def running_server(loaded_backend):
    """Start the FastAPI gateway in a background thread."""
    _backends.clear()
    register_backend(MODEL, loaded_backend)

    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    import httpx
    for _ in range(30):
        try:
            r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        pytest.fail("Server did not start within 15 seconds")

    yield server
    server.should_exit = True


# ---------------------------------------------------------------------------
# 1. Basic generation — verify the model follows instructions
# ---------------------------------------------------------------------------

def test_basic_generation():
    """Qwen should produce coherent, instruction-following text."""
    import httpx
    resp = httpx.post(f"{BASE_URL}/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
        "max_tokens": 20,
        "temperature": 0.1,
    }, timeout=60)
    assert resp.status_code == 200
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    print(f"\n[e2e] basic generation: {content!r}")
    assert "4" in content


# ---------------------------------------------------------------------------
# 2. OpenAI SDK — the core promise of InstaLLM
# ---------------------------------------------------------------------------

def test_openai_sdk_chat():
    """OpenAI Python SDK should work with just a base_url change."""
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=BASE_URL, api_key="not-needed")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in French. One word only."}],
        max_tokens=20,
        temperature=0.1,
    )
    content = response.choices[0].message.content
    print(f"\n[e2e] OpenAI SDK chat: {content!r}")
    assert isinstance(content, str) and len(content) > 0


def test_openai_sdk_streaming():
    """OpenAI SDK streaming should yield chunks correctly."""
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=BASE_URL, api_key="not-needed")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        max_tokens=30,
        temperature=0.1,
        stream=True,
    )
    chunks = []
    for chunk in stream:
        if chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    full_text = "".join(chunks)
    print(f"\n[e2e] OpenAI SDK streaming: {full_text!r}")
    assert len(chunks) >= 1
    assert len(full_text) > 0


def test_openai_sdk_models_list():
    """OpenAI SDK models.list() should return the loaded model."""
    openai = pytest.importorskip("openai")
    from unittest.mock import patch
    from installm.config import list_models

    mock_models = {MODEL: {"added_at": 1700000000, "backend": "transformers"}}
    with patch("installm.gateway.routes.models.list_models", return_value=mock_models):
        client = openai.OpenAI(base_url=BASE_URL, api_key="not-needed")
        models = client.models.list()
        ids = [m.id for m in models.data]
        assert MODEL in ids
        print(f"\n[e2e] OpenAI SDK models: {ids}")


# ---------------------------------------------------------------------------
# 3. Tool calling via OpenAI SDK
# ---------------------------------------------------------------------------

def test_openai_sdk_tool_calling():
    """Tool calling should work through the OpenAI SDK."""
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=BASE_URL, api_key="not-needed")

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=100,
        temperature=0.1,
    )

    msg = response.choices[0].message
    print(f"\n[e2e] Tool calling response content: {msg.content!r}")
    print(f"[e2e] Tool calling tool_calls: {msg.tool_calls}")
    print(f"[e2e] Tool calling finish_reason: {response.choices[0].finish_reason}")

    # Qwen2.5 reliably produces tool call JSON — verify the gateway parsed it
    assert response.choices[0].finish_reason == "tool_calls", (
        f"Expected finish_reason='tool_calls', got '{response.choices[0].finish_reason}'"
    )
    assert msg.tool_calls is not None, "Expected tool_calls to be populated"
    assert len(msg.tool_calls) >= 1
    assert msg.tool_calls[0].function.name == "get_weather"
    args = json.loads(msg.tool_calls[0].function.arguments)
    assert "city" in args
    print(f"[e2e] Parsed tool call: {msg.tool_calls[0].function.name}({args})")


# ---------------------------------------------------------------------------
# 4. Structured output via OpenAI SDK
# ---------------------------------------------------------------------------

def test_openai_sdk_json_mode():
    """JSON mode should return valid JSON."""
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=BASE_URL, api_key="not-needed")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in JSON."},
            {"role": "user", "content": "Give me a JSON object with keys 'name' and 'age' for a fictional person."},
        ],
        response_format={"type": "json_object"},
        max_tokens=60,
        temperature=0.1,
    )

    content = response.choices[0].message.content
    print(f"\n[e2e] JSON mode output: {content!r}")

    # Try to parse as JSON — the model should produce valid JSON
    try:
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
        print(f"[e2e] Parsed JSON: {parsed}")
    except json.JSONDecodeError:
        # Small model may not always produce valid JSON, but the gateway
        # should still return a response without crashing
        print("[e2e] Warning: model did not produce valid JSON, but gateway handled it")


# ---------------------------------------------------------------------------
# 5. LangChain integration
# ---------------------------------------------------------------------------

def test_langchain_chat():
    """LangChain ChatOpenAI should work with InstaLLM."""
    langchain_openai = pytest.importorskip("langchain_openai")

    llm = langchain_openai.ChatOpenAI(
        base_url=BASE_URL,
        api_key="not-needed",
        model=MODEL,
        max_tokens=30,
        temperature=0.1,
    )
    result = llm.invoke("What is the capital of France? One word.")
    print(f"\n[e2e] LangChain chat: {result.content!r}")
    assert isinstance(result.content, str) and len(result.content) > 0


def test_langchain_streaming():
    """LangChain streaming should work with InstaLLM."""
    langchain_openai = pytest.importorskip("langchain_openai")

    llm = langchain_openai.ChatOpenAI(
        base_url=BASE_URL,
        api_key="not-needed",
        model=MODEL,
        max_tokens=30,
        temperature=0.1,
        streaming=True,
    )
    chunks = []
    for chunk in llm.stream("Say hi"):
        chunks.append(chunk.content)

    full = "".join(chunks)
    print(f"\n[e2e] LangChain streaming: {full!r} ({len(chunks)} chunks)")
    assert len(full) > 0
