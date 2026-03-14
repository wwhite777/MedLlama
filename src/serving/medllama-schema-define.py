"""
MedLlama Pydantic schema definitions for API request/response models.

All request and response models used by the FastAPI serving layer
are defined here for consistency and reuse.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for /chat and /chat/stream endpoints."""

    message: str = Field(..., description="User message / medical question")
    use_rag: bool = Field(
        default=True,
        description="Whether to augment the response with RAG retrieval",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability mass",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response (only used by /chat endpoint as a hint)",
    )


class RetrieveRequest(BaseModel):
    """Request body for /retrieve endpoint."""

    query: str = Field(..., description="Query string for document retrieval")
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of documents to retrieve",
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to apply cross-encoder reranking",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """A single retrieved source document."""

    pmid: str = Field(..., description="PubMed ID of the source article")
    title: str = Field(..., description="Title of the source article")
    text: str = Field(..., description="Relevant text snippet")
    score: float = Field(..., description="Retrieval / reranking score")


class ChatResponse(BaseModel):
    """Response body for /chat endpoint."""

    response: str = Field(..., description="Generated response text")
    sources: list[Source] = Field(
        default_factory=list,
        description="Source documents used for RAG context",
    )
    model: str = Field(default="medllama", description="Model identifier")
    usage: dict = Field(
        default_factory=dict,
        description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)",
    )


class RetrieveResponse(BaseModel):
    """Response body for /retrieve endpoint."""

    query: str = Field(..., description="Original query string")
    documents: list[Source] = Field(..., description="Retrieved documents")
    retrieval_time_ms: float = Field(
        ..., description="Retrieval latency in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str = Field(..., description="Service status (ok / degraded / error)")
    model_loaded: bool = Field(..., description="Whether the LLM is loaded and ready")
    qdrant_connected: bool = Field(
        ..., description="Whether Qdrant vector store is reachable"
    )
    gpu_memory_used_mb: float = Field(
        ..., description="GPU memory used in megabytes (0 if no GPU)"
    )
    version: str = Field(..., description="API version string")


# ---------------------------------------------------------------------------
# SSE streaming chunk model (used internally by the streaming endpoint)
# ---------------------------------------------------------------------------

class StreamChunk(BaseModel):
    """Single chunk emitted during SSE streaming."""

    token: str = Field(default="", description="Generated token text")
    finish_reason: str | None = Field(
        default=None,
        description="Reason for finishing (stop, length, error, or None while generating)",
    )
    sources: list[Source] | None = Field(
        default=None,
        description="Sources attached to the final chunk only",
    )
