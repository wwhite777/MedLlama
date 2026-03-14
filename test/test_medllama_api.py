"""
Tests for the MedLlama FastAPI serving endpoints.

Uses httpx AsyncClient with pytest-asyncio to exercise every endpoint
defined in ``medllama-api-serve.py``.
"""

from __future__ import annotations

import importlib
import json
import sys

import httpx
import pytest
import pytest_asyncio

# Import the FastAPI app via the run wrapper
_api = importlib.import_module("src.serving.medllama-api-serve")
app = _api.app

_schemas = importlib.import_module("src.serving.medllama-schema-define")
ChatResponse = _schemas.ChatResponse
RetrieveResponse = _schemas.RetrieveResponse
HealthResponse = _schemas.HealthResponse


@pytest_asyncio.fixture
async def client():
    """Provide an httpx AsyncClient bound to the FastAPI app."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


# ------------------------------------------------------------------ health
@pytest.mark.asyncio
async def test_health_returns_200(client: httpx.AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert "model_loaded" in body
    assert "qdrant_connected" in body
    assert "gpu_memory_used_mb" in body
    assert "version" in body


@pytest.mark.asyncio
async def test_health_matches_schema(client: httpx.AsyncClient):
    resp = await client.get("/health")
    data = resp.json()
    health = HealthResponse(**data)
    assert isinstance(health.gpu_memory_used_mb, float)


# -------------------------------------------------------------------- chat
@pytest.mark.asyncio
async def test_chat_valid_request(client: httpx.AsyncClient):
    payload = {"message": "What are the symptoms of diabetes?"}
    resp = await client.post("/chat", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "response" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)
    assert body["model"] == "medllama"
    # Validate against schema
    ChatResponse(**body)


@pytest.mark.asyncio
async def test_chat_without_rag(client: httpx.AsyncClient):
    payload = {"message": "Hello", "use_rag": False}
    resp = await client.post("/chat", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []


@pytest.mark.asyncio
async def test_chat_custom_params(client: httpx.AsyncClient):
    payload = {
        "message": "Test question",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    resp = await client.post("/chat", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_missing_message_returns_422(client: httpx.AsyncClient):
    resp = await client.post("/chat", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_invalid_temperature_returns_422(client: httpx.AsyncClient):
    payload = {"message": "Hi", "temperature": 5.0}
    resp = await client.post("/chat", json=payload)
    assert resp.status_code == 422


# -------------------------------------------------------------- chat/stream
@pytest.mark.asyncio
async def test_chat_stream_returns_sse(client: httpx.AsyncClient):
    payload = {"message": "Explain hypertension briefly"}
    resp = await client.post("/chat/stream", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Collect all SSE events from the response body
    raw = resp.text
    events = [line for line in raw.split("\n") if line.startswith("data: ")]
    assert len(events) > 0

    # The last "done" event should contain finish_reason="stop"
    last_data_lines = []
    for block in raw.split("\n\n"):
        if "event: done" in block:
            for line in block.split("\n"):
                if line.startswith("data: "):
                    last_data_lines.append(line[len("data: "):])
    assert len(last_data_lines) > 0
    done_payload = json.loads(last_data_lines[0])
    assert done_payload["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_chat_stream_missing_message_returns_422(client: httpx.AsyncClient):
    resp = await client.post("/chat/stream", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------- retrieve
@pytest.mark.asyncio
async def test_retrieve_valid_request(client: httpx.AsyncClient):
    payload = {"query": "diabetes treatment options"}
    resp = await client.post("/retrieve", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == payload["query"]
    assert isinstance(body["documents"], list)
    assert "retrieval_time_ms" in body
    RetrieveResponse(**body)


@pytest.mark.asyncio
async def test_retrieve_custom_top_k(client: httpx.AsyncClient):
    payload = {"query": "heart failure", "top_k": 2}
    resp = await client.post("/retrieve", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["documents"], list)


@pytest.mark.asyncio
async def test_retrieve_missing_query_returns_422(client: httpx.AsyncClient):
    resp = await client.post("/retrieve", json={})
    assert resp.status_code == 422


# ------------------------------------------------------------ OpenAPI docs
@pytest.mark.asyncio
async def test_openapi_schema_available(client: httpx.AsyncClient):
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    assert "/health" in schema["paths"]
    assert "/chat" in schema["paths"]
    assert "/chat/stream" in schema["paths"]
    assert "/retrieve" in schema["paths"]
