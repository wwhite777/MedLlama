#!/usr/bin/env python3
"""
MedLlama Hybrid Retrieval

Implements hybrid search combining dense vector search and BM25 keyword search
via Qdrant, with Reciprocal Rank Fusion (RRF) merging and optional cross-encoder
reranking using BGE-reranker-v2-m3.

Usage:
    from importlib import import_module
    retriever = import_module("src.rag.medllama-hybrid-retrieve")
    engine = retriever.HybridRetriever(config_path="configs/medllama-rag-config.yaml")
    results = engine.retrieve("What is the treatment for hypertension?")
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchText,
)
from rich.console import Console

console = Console()

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "medllama-rag-config.yaml"
)
DEFAULT_QDRANT_LOCAL = (
    Path(__file__).resolve().parents[2] / "data" / "qdrant_local"
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class HybridRetriever:
    """Hybrid dense + BM25 retriever with RRF fusion and cross-encoder reranking."""

    def __init__(
        self,
        config_path: str = str(DEFAULT_CONFIG_PATH),
        qdrant_local_path: Optional[str] = None,
        load_reranker: bool = True,
        load_embedder: bool = True,
    ):
        self.config = load_config(config_path)
        self.qdrant_cfg = self.config["qdrant"]
        self.retrieval_cfg = self.config["retrieval"]
        self.embedding_cfg = self.config["embedding"]

        self.collection_name = self.qdrant_cfg["collection_name"]
        self.dense_top_k = self.retrieval_cfg["dense_top_k"]
        self.bm25_top_k = self.retrieval_cfg["bm25_top_k"]
        self.rrf_k = self.retrieval_cfg["rrf_k"]
        self.final_top_k = self.retrieval_cfg["final_top_k"]

        # Connect to Qdrant
        local_path = qdrant_local_path or str(DEFAULT_QDRANT_LOCAL)
        if Path(local_path).exists():
            self.client = QdrantClient(path=local_path)
        else:
            self.client = QdrantClient(
                host=self.qdrant_cfg["host"],
                port=self.qdrant_cfg["port"],
                timeout=30,
            )

        # Load embedding model for query encoding
        self.embedder = None
        if load_embedder:
            self._load_embedder()

        # Load cross-encoder reranker
        self.reranker = None
        if load_reranker:
            self._load_reranker()

    def _load_embedder(self):
        """Load BGE-M3 for query embedding."""
        device = self.embedding_cfg["device"]
        model_name = self.embedding_cfg["model"]
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(model_name, device=device)
            console.print(f"[green]Embedder loaded: {model_name} on {device}[/green]")
        except Exception as e:
            console.print(f"[yellow]Failed to load embedder: {e}[/yellow]")

    def _load_reranker(self):
        """Load cross-encoder reranker."""
        reranker_cfg = self.retrieval_cfg.get("reranker", {})
        model_name = reranker_cfg.get("model", "BAAI/bge-reranker-v2-m3")
        device = reranker_cfg.get("device", "cpu")
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(model_name, device=device)
            console.print(f"[green]Reranker loaded: {model_name} on {device}[/green]")
        except Exception as e:
            console.print(f"[yellow]Failed to load reranker: {e}[/yellow]")

    def embed_query(self, query: str) -> list[float]:
        """Encode a query string into a dense vector."""
        if self.embedder is None:
            raise RuntimeError("Embedder not loaded")
        embedding = self.embedder.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def dense_search(self, query_vector: list[float], top_k: int = 20) -> list[dict]:
        """Search Qdrant by dense vector similarity."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "pmid": r.payload.get("pmid", ""),
                "title": r.payload.get("title", ""),
                "chunk_text": r.payload.get("chunk_text", ""),
                "abstract": r.payload.get("abstract", ""),
                "journal": r.payload.get("journal", ""),
            }
            for r in results
        ]

    def bm25_search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search Qdrant by BM25 text matching on chunk_text field."""
        # Qdrant text search via scroll with text match filter
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="chunk_text",
                        match=MatchText(text=query),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "id": str(r.id),
                "score": 1.0 / (i + 1),  # Assign rank-based score
                "pmid": r.payload.get("pmid", ""),
                "title": r.payload.get("title", ""),
                "chunk_text": r.payload.get("chunk_text", ""),
                "abstract": r.payload.get("abstract", ""),
                "journal": r.payload.get("journal", ""),
            }
            for i, r in enumerate(results)
        ]

    def rrf_merge(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion to combine dense and BM25 results.

        RRF score = sum(1 / (k + rank_i)) across all rankings where doc appears.
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        merged = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            merged.append(doc)

        return merged

    def rerank(
        self, query: str, documents: list[dict], top_k: int = 5
    ) -> list[dict]:
        """Rerank documents using cross-encoder."""
        if self.reranker is None or not documents:
            return documents[:top_k]

        pairs = [(query, doc["chunk_text"]) for doc in documents]
        rerank_scores = self.reranker.predict(pairs)

        if hasattr(rerank_scores, "tolist"):
            rerank_scores = rerank_scores.tolist()

        for doc, score in zip(documents, rerank_scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
    ) -> dict:
        """
        Full hybrid retrieval pipeline:
        1. Embed query
        2. Dense search
        3. BM25 search
        4. RRF fusion
        5. Cross-encoder reranking (optional)

        Returns dict with documents and timing info.
        """
        top_k = top_k or self.final_top_k
        start_time = time.time()

        # Step 1: Embed query
        query_vector = self.embed_query(query)
        embed_time = time.time() - start_time

        # Step 2: Dense search
        dense_results = self.dense_search(query_vector, self.dense_top_k)
        dense_time = time.time() - start_time - embed_time

        # Step 3: BM25 search
        bm25_results = self.bm25_search(query, self.bm25_top_k)
        bm25_time = time.time() - start_time - embed_time - dense_time

        # Step 4: RRF fusion
        merged = self.rrf_merge(dense_results, bm25_results, k=self.rrf_k)

        # Step 5: Reranking
        if use_reranker and self.reranker is not None:
            # Rerank top candidates
            candidates = merged[: max(top_k * 4, 20)]
            final_docs = self.rerank(query, candidates, top_k)
        else:
            final_docs = merged[:top_k]

        total_time = (time.time() - start_time) * 1000  # ms

        return {
            "query": query,
            "documents": final_docs,
            "retrieval_time_ms": round(total_time, 2),
            "stats": {
                "dense_count": len(dense_results),
                "bm25_count": len(bm25_results),
                "merged_count": len(merged),
                "final_count": len(final_docs),
                "embed_time_ms": round(embed_time * 1000, 2),
                "dense_time_ms": round(dense_time * 1000, 2),
                "bm25_time_ms": round(bm25_time * 1000, 2),
                "reranker_used": use_reranker and self.reranker is not None,
            },
        }


def main():
    """Test the hybrid retriever with sample queries."""
    import argparse

    parser = argparse.ArgumentParser(description="Test MedLlama hybrid retrieval")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--query", type=str, default="What is the treatment for hypertension?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--qdrant-local", type=str, default=None)
    args = parser.parse_args()

    console.print("[bold blue]MedLlama Hybrid Retrieval Test[/bold blue]\n")

    retriever = HybridRetriever(
        config_path=args.config,
        qdrant_local_path=args.qdrant_local,
        load_reranker=not args.no_reranker,
    )

    result = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        use_reranker=not args.no_reranker,
    )

    console.print(f"\n[bold]Query:[/bold] {result['query']}")
    console.print(f"[bold]Retrieval time:[/bold] {result['retrieval_time_ms']} ms")
    console.print(f"[bold]Stats:[/bold] {result['stats']}")
    console.print(f"\n[bold]Top {len(result['documents'])} documents:[/bold]")

    for i, doc in enumerate(result["documents"]):
        score_key = "rerank_score" if "rerank_score" in doc else "rrf_score"
        console.print(f"\n  [{i+1}] PMID: {doc['pmid']} | {score_key}: {doc.get(score_key, 0):.4f}")
        console.print(f"      Title: {doc['title'][:100]}")
        console.print(f"      Text: {doc['chunk_text'][:200]}...")


if __name__ == "__main__":
    main()
