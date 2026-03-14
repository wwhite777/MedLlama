#!/usr/bin/env python3
"""
MedLlama Embedding Generator

Loads PubMed abstracts, chunks them semantically, generates BGE-M3 embeddings,
and saves them for Qdrant ingestion.

Usage:
    CUDA_VISIBLE_DEVICES=2 python src/rag/medllama-embedding-generate.py [--config CONFIG] [--input INPUT]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
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

console = Console()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "medllama-rag-config.yaml"
DEFAULT_INPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "pubmed" / "medllama-pubmed-abstracts.jsonl"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "embeddings" / "medllama-pubmed-embeddings.jsonl"


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def semantic_chunk(
    text: str,
    title: str = "",
    max_chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
) -> list[str]:
    """
    Semantic chunking of abstract text.

    Splits on sentence boundaries, respecting max token size (approximated
    as ~4 chars per token). Prepends title to each chunk for context.
    """
    # Approximate tokens as words (more accurate than chars for English)
    # BGE-M3 uses ~1.3 tokens per word on average
    max_words = int(max_chunk_size * 0.75)  # conservative word estimate
    overlap_words = int(chunk_overlap * 0.75)
    min_words = int(min_chunk_size * 0.75)

    # Combine title + abstract for context
    prefix = f"{title}. " if title else ""

    # Split into sentences
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ".!?" and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    if not sentences:
        return [prefix + text] if len(text.split()) >= min_words else []

    # Build chunks from sentences
    chunks = []
    current_words = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        if current_word_count + sentence_word_count > max_words and current_words:
            chunk_text = prefix + " ".join(current_words)
            chunks.append(chunk_text)

            # Overlap: keep last overlap_words
            if overlap_words > 0:
                overlap_text = current_words[-overlap_words:] if len(current_words) > overlap_words else current_words
                current_words = list(overlap_text)
                current_word_count = len(current_words)
            else:
                current_words = []
                current_word_count = 0

        current_words.extend(sentence_words)
        current_word_count += sentence_word_count

    # Last chunk
    if current_words and current_word_count >= min_words:
        chunk_text = prefix + " ".join(current_words)
        chunks.append(chunk_text)
    elif current_words and not chunks:
        # If this is the only chunk, keep it even if small
        chunk_text = prefix + " ".join(current_words)
        chunks.append(chunk_text)

    return chunks


def load_abstracts(input_path: str) -> list[dict]:
    """Load abstracts from JSONL file."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate BGE-M3 embeddings for MedLlama")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Config YAML path")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Input JSONL path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output embeddings JSONL path")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda:2)")
    args = parser.parse_args()

    config = load_config(args.config)
    emb_cfg = config["embedding"]
    chunk_cfg = config["chunking"]

    model_name = emb_cfg["model"]
    device = args.device or emb_cfg["device"]
    batch_size = args.batch_size or emb_cfg["batch_size"]
    max_length = emb_cfg["max_length"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]MedLlama Embedding Generator[/bold blue]")
    console.print(f"  Model: {model_name}")
    console.print(f"  Device: {device}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Max length: {max_length}")
    console.print(f"  Input: {args.input}")
    console.print(f"  Output: {output_path}")
    console.print()

    # Load abstracts
    console.print("[bold]Loading abstracts...[/bold]")
    records = load_abstracts(args.input)
    console.print(f"  Loaded {len(records)} abstracts")

    # Chunk all abstracts
    console.print("[bold]Chunking abstracts...[/bold]")
    all_chunks = []
    for record in records:
        chunks = semantic_chunk(
            text=record["abstract"],
            title=record.get("title", ""),
            max_chunk_size=chunk_cfg["max_chunk_size"],
            chunk_overlap=chunk_cfg["chunk_overlap"],
            min_chunk_size=chunk_cfg.get("min_chunk_size", 100),
        )
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "pmid": record["pmid"],
                "title": record.get("title", ""),
                "abstract": record["abstract"],
                "chunk_text": chunk_text,
                "chunk_id": i,
                "journal": record.get("journal", ""),
                "pub_date": record.get("pub_date", ""),
                "authors": record.get("authors", []),
            })

    console.print(f"  Generated {len(all_chunks)} chunks from {len(records)} abstracts")
    console.print(f"  Average chunks per abstract: {len(all_chunks) / len(records):.1f}")

    # Load BGE-M3 model
    console.print(f"\n[bold]Loading BGE-M3 model on {device}...[/bold]")
    try:
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(model_name, use_fp16=True, device=device)
        console.print("[green]  Model loaded successfully (FlagEmbedding)[/green]")
        use_flag = True
    except ImportError:
        console.print("[yellow]  FlagEmbedding not available, using sentence-transformers[/yellow]")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        use_flag = False

    # Generate embeddings in batches
    console.print(f"\n[bold]Generating embeddings ({len(all_chunks)} chunks, batch_size={batch_size})...[/bold]")

    chunk_texts = [c["chunk_text"] for c in all_chunks]
    all_embeddings = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total_batches = (len(chunk_texts) + batch_size - 1) // batch_size
        task = progress.add_task("Embedding...", total=total_batches)

        for start in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[start : start + batch_size]

            if use_flag:
                output = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    max_length=max_length,
                )
                # FlagEmbedding returns dict with 'dense_vecs'
                batch_embeddings = output["dense_vecs"]
            else:
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

            if isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = batch_embeddings.cpu().numpy()

            all_embeddings.append(batch_embeddings)
            progress.advance(task)

    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    console.print(f"\n  Embedding shape: {all_embeddings.shape}")
    console.print(f"  Embedding dim: {all_embeddings.shape[1]}")

    # Verify dimension matches config
    expected_dim = config["qdrant"]["vector_size"]
    actual_dim = all_embeddings.shape[1]
    if actual_dim != expected_dim:
        console.print(
            f"[yellow]  Warning: embedding dim {actual_dim} != config vector_size {expected_dim}. "
            f"Will use actual dim {actual_dim}.[/yellow]"
        )

    # Save embeddings with metadata
    console.print(f"\n[bold]Saving embeddings to {output_path}...[/bold]")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk_meta in enumerate(all_chunks):
            record = {
                "pmid": chunk_meta["pmid"],
                "title": chunk_meta["title"],
                "abstract": chunk_meta["abstract"],
                "chunk_text": chunk_meta["chunk_text"],
                "chunk_id": chunk_meta["chunk_id"],
                "journal": chunk_meta["journal"],
                "pub_date": chunk_meta["pub_date"],
                "embedding": all_embeddings[i].tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"[bold green]Saved {len(all_chunks)} embedded chunks ({file_size_mb:.1f} MB)[/bold green]")

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Abstracts: {len(records)}")
    console.print(f"  Chunks: {len(all_chunks)}")
    console.print(f"  Embedding dim: {actual_dim}")
    console.print(f"  Output file: {output_path}")


if __name__ == "__main__":
    main()
