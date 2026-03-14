#!/usr/bin/env python3
"""
MedLlama Demo Output Generator

Generates sample outputs for the README and portfolio by running RAG retrieval
on representative medical questions. Works without GPU -- only the retrieval
pipeline is exercised. Model responses are left as placeholders to be filled
in when a GPU is available.

Usage:
    python3 src/serving/medllama-demo-generate.py

Output:
    result/demo/medllama-demo-outputs.md
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Demo questions spanning diverse medical topics
# ---------------------------------------------------------------------------
DEMO_QUESTIONS: list[dict[str, str]] = [
    {
        "topic": "Cardiology",
        "question": "What are the current guideline-recommended first-line treatments for heart failure with reduced ejection fraction?",
    },
    {
        "topic": "Pharmacology",
        "question": "What is the mechanism of action of metformin and its common side effects in type 2 diabetes management?",
    },
    {
        "topic": "Diagnosis",
        "question": "How is pulmonary embolism diagnosed, and what are the key clinical decision rules used in emergency settings?",
    },
    {
        "topic": "Oncology",
        "question": "What are the latest advances in immune checkpoint inhibitor therapy for non-small cell lung cancer?",
    },
    {
        "topic": "Infectious Disease",
        "question": "What is the recommended empiric antibiotic therapy for community-acquired pneumonia in hospitalized adults?",
    },
    {
        "topic": "Neurology",
        "question": "What are the disease-modifying therapies available for relapsing-remitting multiple sclerosis?",
    },
    {
        "topic": "Endocrinology",
        "question": "How should diabetic ketoacidosis be managed in the emergency department, and what are the key monitoring parameters?",
    },
    {
        "topic": "Nephrology",
        "question": "What are the indications for initiating dialysis in patients with chronic kidney disease?",
    },
    {
        "topic": "Gastroenterology",
        "question": "What is the current evidence for fecal microbiota transplantation in treating recurrent Clostridioides difficile infection?",
    },
    {
        "topic": "Psychiatry",
        "question": "What are the first-line pharmacological treatments for major depressive disorder, and how do SSRIs compare to SNRIs?",
    },
]

SYSTEM_PROMPT_PREVIEW = (
    "You are MedLlama, a knowledgeable medical assistant trained on medical "
    "literature. You provide accurate, evidence-based medical information."
)


def try_load_retriever():
    """Attempt to load the RAG retriever. Returns None on failure."""
    try:
        rag_config = str(PROJECT_ROOT / "configs" / "medllama-rag-config.yaml")
        qdrant_path = PROJECT_ROOT / "data" / "qdrant_local"

        retriever_mod = importlib.import_module("src.rag.medllama-hybrid-retrieve")

        retriever = retriever_mod.HybridRetriever(
            config_path=rag_config,
            qdrant_local_path=str(qdrant_path) if qdrant_path.exists() else None,
            load_reranker=True,
            load_embedder=True,
        )
        return retriever
    except Exception as e:
        print(f"[WARN] Could not load retriever: {e}")
        return None


def format_source_block(doc: dict, index: int) -> str:
    """Format a single retrieved source for the markdown output."""
    pmid = doc.get("pmid", "N/A")
    title = doc.get("title", "Untitled")
    text = doc.get("chunk_text", "")[:300]
    score_key = "rerank_score" if "rerank_score" in doc else "rrf_score"
    score = doc.get(score_key, 0.0)

    return (
        f"  **[{index}]** PMID: {pmid} | Score: {score:.4f}\n"
        f"  *{title}*\n"
        f"  > {text}...\n"
    )


def build_rag_prompt_preview(question: str, sources: list[dict]) -> str:
    """Build a preview of the RAG-augmented prompt that would be sent to the LLM."""
    context_parts = []
    for i, doc in enumerate(sources):
        pmid = doc.get("pmid", "N/A")
        title = doc.get("title", "Untitled")
        text = doc.get("chunk_text", "")[:200]
        context_parts.append(
            f"[Source {i+1}] PMID: {pmid}\nTitle: {title}\n{text}..."
        )
    context_str = "\n\n".join(context_parts)

    prompt = textwrap.dedent(f"""\
        **System**: {SYSTEM_PROMPT_PREVIEW}

        **User**: Based on the following medical literature, answer the user's question.

        === Retrieved Medical Literature ===
        {context_str}
        === End of Retrieved Literature ===

        Question: {question}""")
    return prompt


def generate_demo_outputs(retriever) -> str:
    """Generate the full demo markdown document."""
    lines: list[str] = []
    lines.append("# MedLlama Demo Outputs\n")
    lines.append(
        "Generated by `src/serving/medllama-demo-generate.py` on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.\n"
    )
    lines.append(
        "This document shows RAG retrieval results for 10 representative medical "
        "questions. Model responses are placeholders -- run with GPU to generate "
        "actual completions.\n"
    )
    lines.append("---\n")

    for i, item in enumerate(DEMO_QUESTIONS, 1):
        topic = item["topic"]
        question = item["question"]

        lines.append(f"## {i}. {topic}\n")
        lines.append(f"**Query**: {question}\n")

        # Attempt retrieval
        if retriever is not None:
            try:
                t0 = time.time()
                result = retriever.retrieve(
                    query=question,
                    top_k=5,
                    use_reranker=True,
                )
                elapsed = (time.time() - t0) * 1000
                docs = result["documents"]
                stats = result.get("stats", {})

                lines.append(
                    f"**Retrieval**: {len(docs)} documents in {elapsed:.0f} ms "
                    f"(dense: {stats.get('dense_count', '?')}, "
                    f"BM25: {stats.get('bm25_count', '?')}, "
                    f"reranker: {stats.get('reranker_used', '?')})\n"
                )
                lines.append("### Retrieved Sources\n")
                for j, doc in enumerate(docs, 1):
                    lines.append(format_source_block(doc, j))

                lines.append("### Formatted Prompt (preview)\n")
                lines.append("```")
                lines.append(build_rag_prompt_preview(question, docs))
                lines.append("```\n")

            except Exception as e:
                lines.append(f"**Retrieval failed**: {e}\n")
        else:
            lines.append(
                "**Retrieval**: Skipped (Qdrant not available or embedder not loaded)\n"
            )

        lines.append("### Model Response\n")
        lines.append(
            "*[Placeholder -- run with GPU to generate actual model response]*\n"
        )
        lines.append("---\n")

    # Summary footer
    lines.append("## Summary\n")
    lines.append("| # | Topic | Question (truncated) | Sources Retrieved |")
    lines.append("|---|---|---|---|")
    for i, item in enumerate(DEMO_QUESTIONS, 1):
        q_short = item["question"][:60] + "..."
        lines.append(f"| {i} | {item['topic']} | {q_short} | 5 |")
    lines.append("")

    return "\n".join(lines)


def main():
    print("MedLlama Demo Output Generator")
    print("=" * 40)

    # Try to load retriever
    print("Loading RAG retriever...")
    retriever = try_load_retriever()
    if retriever is not None:
        print("Retriever loaded successfully.")
    else:
        print("Retriever not available. Will generate output with placeholders.")

    # Generate outputs
    print(f"Generating demo outputs for {len(DEMO_QUESTIONS)} questions...")
    output = generate_demo_outputs(retriever)

    # Save
    output_dir = PROJECT_ROOT / "result" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "medllama-demo-outputs.md"
    output_path.write_text(output, encoding="utf-8")
    print(f"Demo outputs saved to {output_path}")
    print(f"Output size: {len(output)} chars, {output.count(chr(10))} lines")

    return str(output_path)


if __name__ == "__main__":
    main()
