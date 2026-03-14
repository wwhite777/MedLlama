#!/usr/bin/env python3
"""
MedLlama RAG Retrieval Evaluation

Evaluates the hybrid retrieval pipeline quality using MedQA questions as test queries.
Measures recall@5, retrieval time, and BM25 vs dense contribution.

Usage:
    CUDA_VISIBLE_DEVICES=2 python src/eval/medllama-rag-evaluate.py --num-queries 50
"""

import argparse
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "medllama-rag-config.yaml"
QDRANT_LOCAL_PATH = PROJECT_ROOT / "data" / "qdrant_local"
RESULT_PATH = PROJECT_ROOT / "result" / "eval" / "medllama-rag-results.json"


def extract_medical_terms(text: str) -> set[str]:
    """Extract meaningful medical terms from text (longer words, lowered)."""
    stopwords = {
        "what", "which", "that", "this", "these", "those", "with", "from",
        "have", "been", "were", "does", "will", "would", "could", "should",
        "most", "more", "than", "about", "into", "over", "after", "before",
        "during", "between", "through", "following", "associated", "likely",
        "patient", "patients", "year", "years", "case", "male", "female",
        "diagnosis", "treatment", "condition", "common", "finding", "findings",
        "cause", "present", "presents", "presented", "history", "exam",
        "examination", "result", "results", "test", "type", "form",
        "also", "used", "using", "best", "next", "step", "first",
        "include", "includes", "including", "among", "woman", "child",
        "hospital", "physician", "doctor", "nurse", "clinic",
    }
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in stopwords}


def check_relevance(query: str, answer_text: str, documents: list[dict]) -> bool:
    """
    Check if any retrieved document is relevant to the query+answer.
    Relevant = document text contains at least 2 key medical terms
    from the question and/or answer.
    """
    query_terms = extract_medical_terms(query)
    answer_terms = extract_medical_terms(answer_text)
    key_terms = query_terms | answer_terms

    if len(key_terms) < 2:
        # Too few terms to judge, be lenient
        return len(documents) > 0

    for doc in documents:
        doc_text = doc.get("chunk_text", "").lower()
        matches = sum(1 for term in key_terms if term in doc_text)
        if matches >= 2:
            return True

    return False


def main():
    parser = argparse.ArgumentParser(description="MedLlama RAG Retrieval Evaluation")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of test queries")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="RAG config path")
    parser.add_argument("--qdrant-local", type=str, default=str(QDRANT_LOCAL_PATH))
    parser.add_argument("--output", type=str, default=str(RESULT_PATH))
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "offline"

    console.print("[bold blue]MedLlama RAG Retrieval Evaluation[/bold blue]")
    console.print(f"  Queries: {args.num_queries}")
    console.print(f"  Top-K: {args.top_k}")
    console.print(f"  Config: {args.config}")
    console.print(f"  Qdrant: {args.qdrant_local}")
    console.print()

    # Load test queries from MedQA
    console.print("[bold]Loading MedQA test questions as queries...[/bold]")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    num_queries = min(args.num_queries, len(dataset))
    console.print(f"  Using {num_queries} queries\n")

    # Load the hybrid retriever
    console.print("[bold]Loading hybrid retriever...[/bold]")
    sys.path.insert(0, str(PROJECT_ROOT))
    retriever_mod = importlib.import_module("src.rag.medllama-hybrid-retrieve")
    retriever = retriever_mod.HybridRetriever(
        config_path=args.config,
        qdrant_local_path=args.qdrant_local,
        load_reranker=False,  # Skip reranker for speed
        load_embedder=True,
    )
    console.print("[green]Retriever loaded[/green]\n")

    # Run retrieval evaluation
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    retrieval_times = []
    dense_counts = []
    bm25_counts = []
    relevant_count = 0
    per_query_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Retrieving", total=num_queries)

        for i in range(num_queries):
            sample = dataset[i]
            query = sample["question"]
            options = sample["options"]
            answer_idx = sample["answer_idx"]

            # Get the correct answer text for relevance checking
            if isinstance(answer_idx, int):
                gt_letter = idx_to_letter.get(answer_idx, "A")
            else:
                gt_letter = str(answer_idx).strip()
            answer_text = options.get(gt_letter, "")

            # Run retrieval
            start = time.time()
            result = retriever.retrieve(
                query=query,
                top_k=args.top_k,
                use_reranker=False,
            )
            elapsed_ms = (time.time() - start) * 1000

            documents = result["documents"]
            stats = result["stats"]

            retrieval_times.append(elapsed_ms)
            dense_counts.append(stats["dense_count"])
            bm25_counts.append(stats["bm25_count"])

            # Check relevance
            is_relevant = check_relevance(query, answer_text, documents)
            if is_relevant:
                relevant_count += 1

            per_query_results.append({
                "index": i,
                "query": query[:200],
                "retrieval_time_ms": round(elapsed_ms, 2),
                "num_docs_returned": len(documents),
                "dense_count": stats["dense_count"],
                "bm25_count": stats["bm25_count"],
                "relevant": is_relevant,
                "top_doc_preview": documents[0]["chunk_text"][:150] if documents else "",
            })

            progress.advance(task)

    # Compute metrics
    recall_at_5 = relevant_count / num_queries if num_queries > 0 else 0
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
    median_retrieval_time = sorted(retrieval_times)[len(retrieval_times) // 2] if retrieval_times else 0
    avg_dense_count = sum(dense_counts) / len(dense_counts) if dense_counts else 0
    avg_bm25_count = sum(bm25_counts) / len(bm25_counts) if bm25_counts else 0

    # BM25 vs Dense contribution analysis
    both_contributed = sum(1 for d, b in zip(dense_counts, bm25_counts) if d > 0 and b > 0)
    dense_only = sum(1 for d, b in zip(dense_counts, bm25_counts) if d > 0 and b == 0)
    bm25_only = sum(1 for d, b in zip(dense_counts, bm25_counts) if d == 0 and b > 0)
    neither = sum(1 for d, b in zip(dense_counts, bm25_counts) if d == 0 and b == 0)

    output = {
        "eval_type": "rag_retrieval",
        "timestamp": datetime.now().isoformat(),
        "num_queries": num_queries,
        "top_k": args.top_k,
        "metrics": {
            "recall_at_5": round(recall_at_5, 4),
            "relevant_queries": relevant_count,
            "avg_retrieval_time_ms": round(avg_retrieval_time, 2),
            "median_retrieval_time_ms": round(median_retrieval_time, 2),
            "min_retrieval_time_ms": round(min(retrieval_times), 2) if retrieval_times else 0,
            "max_retrieval_time_ms": round(max(retrieval_times), 2) if retrieval_times else 0,
        },
        "contribution_analysis": {
            "avg_dense_results": round(avg_dense_count, 1),
            "avg_bm25_results": round(avg_bm25_count, 1),
            "both_contributed": both_contributed,
            "dense_only": dense_only,
            "bm25_only": bm25_only,
            "neither": neither,
        },
        "per_query_results": per_query_results,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")

    # Print summary
    table = Table(title="RAG Retrieval Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Recall@5", f"{recall_at_5:.4f}")
    table.add_row("Relevant queries", f"{relevant_count}/{num_queries}")
    table.add_row("Avg retrieval time", f"{avg_retrieval_time:.1f} ms")
    table.add_row("Median retrieval time", f"{median_retrieval_time:.1f} ms")
    table.add_row("Avg dense results", f"{avg_dense_count:.1f}")
    table.add_row("Avg BM25 results", f"{avg_bm25_count:.1f}")
    table.add_row("Both contributed", f"{both_contributed}/{num_queries}")
    table.add_row("Dense only", f"{dense_only}/{num_queries}")
    table.add_row("BM25 only", f"{bm25_only}/{num_queries}")
    console.print(table)

    # W&B logging
    try:
        import wandb
        wandb.init(
            project="medllama-eval",
            name=f"rag-eval-{num_queries}",
            config={"num_queries": num_queries, "top_k": args.top_k},
        )
        wandb.log({
            "recall_at_5": recall_at_5,
            "avg_retrieval_time_ms": avg_retrieval_time,
            "avg_dense_results": avg_dense_count,
            "avg_bm25_results": avg_bm25_count,
        })
        wandb.finish()
        console.print("[green]Logged to W&B (offline)[/green]")
    except Exception as e:
        console.print(f"[yellow]W&B logging failed: {e}[/yellow]")


if __name__ == "__main__":
    main()
