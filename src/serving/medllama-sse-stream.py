"""
MedLlama SSE (Server-Sent Events) streaming utilities.

Provides helper functions for formatting token-level streaming responses
as SSE events conforming to the text/event-stream content type.
"""

from __future__ import annotations

import json
import asyncio
from typing import AsyncIterator

# We import schemas via importlib because the filename contains hyphens.
import importlib
_schemas = importlib.import_module("src.serving.medllama-schema-define")
Source = _schemas.Source
StreamChunk = _schemas.StreamChunk


# ---------------------------------------------------------------------------
# SSE formatting helpers
# ---------------------------------------------------------------------------

def format_sse_event(data: str, event: str | None = None) -> str:
    """Format a single SSE event string.

    Parameters
    ----------
    data : str
        JSON-serialised payload.
    event : str | None
        Optional SSE event name (e.g. ``"token"``, ``"done"``, ``"error"``).

    Returns
    -------
    str
        Properly formatted SSE event ending with a double newline.
    """
    lines: list[str] = []
    if event is not None:
        lines.append(f"event: {event}")
    # SSE spec: multi-line data must have each line prefixed with "data: "
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")  # trailing blank line to delimit the event
    lines.append("")
    return "\n".join(lines)


def _chunk_to_sse(chunk: StreamChunk, event: str | None = None) -> str:
    """Serialize a StreamChunk to an SSE event string."""
    return format_sse_event(chunk.model_dump_json(), event=event)


# ---------------------------------------------------------------------------
# Token-level SSE event constructors
# ---------------------------------------------------------------------------

def token_event(token: str) -> str:
    """Create an SSE event for a single generated token."""
    chunk = StreamChunk(token=token)
    return _chunk_to_sse(chunk, event="token")


def done_event(sources: list[Source] | None = None) -> str:
    """Create the final SSE event signalling generation is complete.

    Optionally includes the retrieved source documents.
    """
    chunk = StreamChunk(
        token="",
        finish_reason="stop",
        sources=sources,
    )
    return _chunk_to_sse(chunk, event="done")


def error_event(message: str) -> str:
    """Create an SSE event for an error during generation."""
    chunk = StreamChunk(
        token="",
        finish_reason="error",
    )
    payload = {"error": message, "chunk": chunk.model_dump()}
    return format_sse_event(json.dumps(payload), event="error")


# ---------------------------------------------------------------------------
# Placeholder async generator (will be replaced by real vLLM streaming)
# ---------------------------------------------------------------------------

async def placeholder_stream(
    message: str,
    use_rag: bool = True,
) -> AsyncIterator[str]:
    """Yield SSE events for a placeholder (stub) streaming response.

    This simulates token-by-token generation. It will be replaced by
    a real vLLM-backed generator on Day 3.
    """
    placeholder_text = (
        f"This is a placeholder streaming response to: \"{message}\". "
        "The real model is not loaded yet. "
        "Once the vLLM backend is connected, this will produce "
        "actual medical answers with citation support."
    )

    # Emit tokens word-by-word to simulate streaming
    words = placeholder_text.split(" ")
    for i, word in enumerate(words):
        tok = word if i == 0 else f" {word}"
        yield token_event(tok)
        await asyncio.sleep(0.02)  # simulate generation latency

    # Build placeholder sources if RAG was requested
    sources: list[Source] | None = None
    if use_rag:
        sources = [
            Source(
                pmid="00000001",
                title="Placeholder Source Article",
                text="This is a placeholder source document.",
                score=0.95,
            ),
        ]

    yield done_event(sources=sources)
