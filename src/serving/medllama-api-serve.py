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

import asyncio
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

format_sse_event = _sse.format_sse_event
token_event = _sse.token_event
done_event = _sse.done_event
error_event = _sse.error_event
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
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "medllama-serving-config.yaml"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f)
    logger.warning("Config file not found at %s -- using defaults", _CONFIG_PATH)
    return {}


CONFIG: dict = _load_config()
API_CFG: dict = CONFIG.get("api", {})
GEN_CFG: dict = CONFIG.get("generation", {})
RAG_CFG: dict = CONFIG.get("rag", {})
VLLM_CFG: dict = CONFIG.get("vllm", {})

# ---------------------------------------------------------------------------
# Application state (populated during lifespan startup)
# ---------------------------------------------------------------------------
_state: dict = {
    "model_loaded": False,
    "qdrant_connected": False,
    "llm": None,           # vLLM AsyncLLMEngine or OpenAI client
    "tokenizer": None,
    "rag_orchestrator": None,
}


def _try_load_model():
    """Load the merged model via vLLM or transformers as fallback."""
    model_path = VLLM_CFG.get("model", "checkpoints/merged")
    full_model_path = _PROJECT_ROOT / model_path

    if not full_model_path.exists():
        logger.warning("Model path %s not found, trying as HF model ID", model_path)
        full_model_path = model_path

    # Try vLLM first
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=str(full_model_path),
            dtype=VLLM_CFG.get("dtype", "bfloat16"),
            gpu_memory_utilization=VLLM_CFG.get("gpu_memory_utilization", 0.9),
            max_model_len=VLLM_CFG.get("max_model_len", 4096),
            trust_remote_code=True,
        )
        _state["llm"] = llm
        _state["llm_type"] = "vllm"
        _state["model_loaded"] = True
        logger.info("Model loaded via vLLM: %s", full_model_path)
        return
    except Exception as e:
        logger.warning("vLLM failed: %s. Trying transformers fallback.", e)

    # Fallback: transformers pipeline
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(
            str(full_model_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(full_model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _state["llm"] = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        _state["tokenizer"] = tokenizer
        _state["llm_type"] = "transformers"
        _state["model_loaded"] = True
        logger.info("Model loaded via transformers: %s", full_model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        _state["model_loaded"] = False


def _try_load_rag():
    """Load the RAG orchestrator."""
    try:
        rag_mod = importlib.import_module("src.rag.medllama-rag-orchestrate")
        rag_config = str(_PROJECT_ROOT / "configs" / "medllama-rag-config.yaml")
        _state["rag_orchestrator"] = rag_mod.RAGOrchestrator(
            config_path=rag_config,
            load_models=True,
        )
        _state["qdrant_connected"] = True
        logger.info("RAG orchestrator loaded")
    except Exception as e:
        logger.warning("RAG orchestrator failed to load: %s", e)
        _state["qdrant_connected"] = False


def _generate_response(messages: list[dict], max_tokens: int = 1024,
                        temperature: float = 0.3, top_p: float = 0.9) -> str:
    """Generate a response using the loaded LLM."""
    llm = _state.get("llm")
    if llm is None:
        return "Model not loaded. Please check /health endpoint."

    llm_type = _state.get("llm_type", "")

    if llm_type == "vllm":
        from vllm import SamplingParams
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    elif llm_type == "transformers":
        pipe = llm
        tokenizer = _state["tokenizer"]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        result = pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            return_full_text=False,
        )
        return result[0]["generated_text"]

    return "Unknown LLM backend."


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting MedLlama API server ...")

    # Load RAG first (doesn't need GPU model)
    _try_load_rag()

    # Load model
    _try_load_model()

    logger.info(
        "Startup complete. model_loaded=%s, qdrant_connected=%s",
        _state["model_loaded"],
        _state["qdrant_connected"],
    )
    yield
    logger.info("Shutting down MedLlama API server ...")
    _state["llm"] = None
    _state["rag_orchestrator"] = None
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s  -> %d  (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------
def _gpu_memory_used_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            total = 0.0
            for i in range(torch.cuda.device_count()):
                total += torch.cuda.memory_allocated(i) / (1024 * 1024)
            return round(total, 1)
    except Exception:
        pass
    return 0.0


# ========================== ENDPOINTS ======================================


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state["model_loaded"] else "degraded",
        model_loaded=_state["model_loaded"],
        qdrant_connected=_state["qdrant_connected"],
        gpu_memory_used_mb=_gpu_memory_used_mb(),
        version=API_CFG.get("version", "0.1.0"),
    )


@app.post("/chat", response_model=ChatResponse, tags=["inference"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Generate a chat response with optional RAG retrieval."""
    logger.info("POST /chat  message=%r  use_rag=%s", request.message[:80], request.use_rag)
    start = time.perf_counter()

    sources = []
    rag_orch = _state.get("rag_orchestrator")

    # RAG retrieval
    if request.use_rag and rag_orch is not None:
        rag_result = rag_orch.run(
            query=request.message,
            use_rag=True,
        )
        messages = rag_result["messages"]
        sources = [
            Source(pmid=s["pmid"], title=s["title"], text=s["text"], score=s["score"])
            for s in rag_result["sources"]
        ]
    else:
        # No RAG — simple message
        rag_mod = importlib.import_module("src.rag.medllama-rag-orchestrate")
        messages = [
            {"role": "system", "content": rag_mod.SYSTEM_PROMPT},
            {"role": "user", "content": request.message},
        ]

    # Generate
    if _state["model_loaded"]:
        response_text = _generate_response(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    else:
        response_text = (
            f"[Model not loaded] Placeholder response to: \"{request.message}\". "
            "Check /health endpoint for model status."
        )

    elapsed = time.perf_counter() - start
    return ChatResponse(
        response=response_text,
        sources=sources,
        model="medllama",
        usage={
            "prompt_tokens": sum(len(m["content"].split()) for m in messages),
            "completion_tokens": len(response_text.split()),
            "total_tokens": sum(len(m["content"].split()) for m in messages) + len(response_text.split()),
            "latency_ms": round(elapsed * 1000, 1),
        },
    )


@app.post("/chat/stream", tags=["inference"])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a chat response as Server-Sent Events."""
    logger.info("POST /chat/stream  message=%r  use_rag=%s", request.message[:80], request.use_rag)

    async def generate_stream():
        rag_orch = _state.get("rag_orchestrator")
        sources = []

        # RAG retrieval
        if request.use_rag and rag_orch is not None:
            rag_result = rag_orch.run(query=request.message, use_rag=True)
            messages = rag_result["messages"]
            sources = [
                Source(pmid=s["pmid"], title=s["title"], text=s["text"], score=s["score"])
                for s in rag_result["sources"]
            ]
        else:
            rag_mod = importlib.import_module("src.rag.medllama-rag-orchestrate")
            messages = [
                {"role": "system", "content": rag_mod.SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ]

        if _state["model_loaded"] and _state.get("llm_type") == "vllm":
            # vLLM streaming not available in sync mode, generate full then stream
            response_text = _generate_response(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            # Stream word by word
            words = response_text.split(" ")
            for i, word in enumerate(words):
                tok = word if i == 0 else f" {word}"
                yield token_event(tok)
                await asyncio.sleep(0.01)
        elif _state["model_loaded"]:
            response_text = _generate_response(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            words = response_text.split(" ")
            for i, word in enumerate(words):
                tok = word if i == 0 else f" {word}"
                yield token_event(tok)
                await asyncio.sleep(0.01)
        else:
            async for event in placeholder_stream(request.message, request.use_rag):
                yield event
            return

        yield done_event(sources=sources if sources else None)

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Retrieve relevant documents from the vector store."""
    logger.info("POST /retrieve  query=%r  top_k=%d", request.query[:80], request.top_k)

    rag_orch = _state.get("rag_orchestrator")

    if rag_orch is not None:
        result = rag_orch.retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )
        docs = [
            Source(
                pmid=d.get("pmid", ""),
                title=d.get("title", ""),
                text=d.get("chunk_text", "")[:500],
                score=round(d.get("rerank_score", d.get("rrf_score", 0)), 4),
            )
            for d in result["documents"]
        ]
        return RetrieveResponse(
            query=request.query,
            documents=docs,
            retrieval_time_ms=result["retrieval_time_ms"],
        )
    else:
        # Placeholder if RAG not loaded
        return RetrieveResponse(
            query=request.query,
            documents=[],
            retrieval_time_ms=0,
        )
