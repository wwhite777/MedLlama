"""
MedLlama FastAPI application -- main serving entry point.

Endpoints
---------
- GET  /health       -- Health check (model status, Qdrant, GPU memory)
- POST /chat         -- Chat completion (optional RAG)
- POST /chat/stream  -- Streaming chat via SSE
- POST /retrieve     -- Direct document retrieval from Qdrant
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Import sibling modules (hyphenated filenames require importlib)
# ---------------------------------------------------------------------------
_schemas = importlib.import_module("src.serving.medllama-schema-define")
_sse = importlib.import_module("src.serving.medllama-sse-stream")

ChatRequest = _schemas.ChatRequest
ChatResponse = _schemas.ChatResponse
RetrieveRequest = _schemas.RetrieveRequest
RetrieveResponse = _schemas.RetrieveResponse
HealthResponse = _schemas.HealthResponse
Source = _schemas.Source

placeholder_stream = _sse.placeholder_stream

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("medllama.api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "medllama-serving-config.yaml"


def _load_config() -> dict:
    """Load the serving configuration YAML."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f)
    logger.warning("Config file not found at %s -- using defaults", _CONFIG_PATH)
    return {}


CONFIG: dict = _load_config()
API_CFG: dict = CONFIG.get("api", {})
GEN_CFG: dict = CONFIG.get("generation", {})
RAG_CFG: dict = CONFIG.get("rag", {})

# ---------------------------------------------------------------------------
# Application state (populated during lifespan startup)
# ---------------------------------------------------------------------------
_state: dict = {
    "model_loaded": False,
    "qdrant_connected": False,
}


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown hooks."""
    logger.info("Starting MedLlama API server ...")
    # TODO (Day 3): load vLLM engine and connect to Qdrant here.
    _state["model_loaded"] = False
    _state["qdrant_connected"] = False
    logger.info("Startup complete (model and Qdrant placeholders only).")
    yield
    logger.info("Shutting down MedLlama API server ...")
    # TODO (Day 3): clean up vLLM engine and Qdrant client here.
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title=API_CFG.get("title", "MedLlama API"),
    version=API_CFG.get("version", "0.1.0"),
    description="Healthcare LLM API with RAG-augmented medical question answering.",
    lifespan=lifespan,
)

# -- CORS middleware --------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Request logging middleware ---------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request with timing information."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s  -> %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# -- Global exception handler -----------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler returning a JSON error body."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------
def _gpu_memory_used_mb() -> float:
    """Return GPU memory used in MB, or 0.0 if CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            # Sum across all visible devices
            total = 0.0
            for i in range(torch.cuda.device_count()):
                total += torch.cuda.memory_allocated(i) / (1024 * 1024)
            return round(total, 1)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Placeholder source generator
# ---------------------------------------------------------------------------
def _placeholder_sources(n: int = 3) -> list[Source]:
    """Return placeholder source documents for stub endpoints."""
    return [
        Source(
            pmid=f"0000000{i}",
            title=f"Placeholder Medical Article {i}",
            text=f"This is placeholder text for source document {i}.",
            score=round(0.95 - i * 0.05, 2),
        )
        for i in range(1, n + 1)
    ]


# ========================== ENDPOINTS ======================================


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Return current health status of the API and its dependencies."""
    return HealthResponse(
        status="ok" if _state["model_loaded"] else "degraded",
        model_loaded=_state["model_loaded"],
        qdrant_connected=_state["qdrant_connected"],
        gpu_memory_used_mb=_gpu_memory_used_mb(),
        version=API_CFG.get("version", "0.1.0"),
    )


@app.post("/chat", response_model=ChatResponse, tags=["inference"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Generate a (non-streaming) chat response.

    When the real model is loaded this will invoke vLLM and optionally
    perform RAG retrieval. For now it returns a placeholder response.
    """
    logger.info("POST /chat  message=%r  use_rag=%s", request.message[:80], request.use_rag)

    # Placeholder response -- will be wired to vLLM + RAG on Day 3
    sources = _placeholder_sources() if request.use_rag else []
    response_text = (
        f"This is a placeholder response to: \"{request.message}\". "
        "The real model is not loaded yet."
    )

    return ChatResponse(
        response=response_text,
        sources=sources,
        model="medllama",
        usage={
            "prompt_tokens": len(request.message.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(request.message.split()) + len(response_text.split()),
        },
    )


@app.post("/chat/stream", tags=["inference"])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a chat response as Server-Sent Events (SSE).

    Each event contains a JSON-serialised ``StreamChunk``. The final event
    has ``finish_reason="stop"`` and optionally includes source documents.
    """
    logger.info("POST /chat/stream  message=%r  use_rag=%s", request.message[:80], request.use_rag)

    return StreamingResponse(
        placeholder_stream(message=request.message, use_rag=request.use_rag),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering if present
        },
    )


@app.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Retrieve relevant documents from the vector store.

    This will be connected to Qdrant on Day 3. Currently returns
    placeholder documents.
    """
    logger.info("POST /retrieve  query=%r  top_k=%d", request.query[:80], request.top_k)

    start = time.perf_counter()
    docs = _placeholder_sources(n=request.top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return RetrieveResponse(
        query=request.query,
        documents=docs,
        retrieval_time_ms=round(elapsed_ms, 2),
    )
