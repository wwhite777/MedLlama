#!/usr/bin/env python3
"""
MedLlama Qdrant Ingestion

Loads embedded chunks and upserts them into Qdrant vector database.
Creates the collection with proper configuration for hybrid search
(dense vectors + BM25 text index).

Usage:
    python src/rag/medllama-qdrant-store.py [--config CONFIG] [--input INPUT]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)
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
DEFAULT_INPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "embeddings" / "medllama-pubmed-embeddings.jsonl"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_qdrant_running(host: str, port: int) -> bool:
    """Check if Qdrant is running, attempt to start if not."""
    client = QdrantClient(host=host, port=port, timeout=5)
    try:
        client.get_collections()
        console.print("[green]  Qdrant is running[/green]")
        return True
    except Exception:
        console.print("[yellow]  Qdrant not reachable. Attempting to start via Docker...[/yellow]")
        try:
            # Check if container exists but is stopped
            result = subprocess.run(
                ["docker", "start", "qdrant"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                time.sleep(3)
                try:
                    client.get_collections()
                    console.print("[green]  Qdrant started (existing container)[/green]")
                    return True
                except Exception:
                    pass

            # Start new container
            storage_path = Path(__file__).resolve().parents[2] / "data" / "qdrant_storage"
            storage_path.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", "qdrant",
                    "-p", f"{port}:{port}",
                    "-p", "6334:6334",
                    "-v", f"{storage_path}:/qdrant/storage",
                    "qdrant/qdrant:latest",
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                console.print("[green]  Qdrant container started[/green]")
                time.sleep(5)
                try:
                    client.get_collections()
                    return True
                except Exception:
                    pass
        except Exception as e:
            console.print(f"[red]  Failed to start Qdrant: {e}[/red]")

        console.print("[red]  Could not connect to Qdrant. Please start it manually:[/red]")
        console.print("    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 "
                       "-v /home/wjeong/ml/medllama/data/qdrant_storage:/qdrant/storage "
                       "qdrant/qdrant:latest")
        return False


def load_embeddings(input_path: str) -> list[dict]:
    """Load embedded chunks from JSONL."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: str,
    hnsw_config: dict,
) -> None:
    """Create or recreate the Qdrant collection."""
    dist_map = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot": Distance.DOT,
    }
    qdrant_distance = dist_map.get(distance, Distance.COSINE)

    # Check if collection exists
    try:
        existing = client.get_collection(collection_name)
        console.print(f"[yellow]  Collection '{collection_name}' exists with "
                       f"{existing.points_count} points. Recreating...[/yellow]")
        client.delete_collection(collection_name)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=qdrant_distance,
            hnsw_config=HnswConfigDiff(
                m=hnsw_config.get("m", 16),
                ef_construct=hnsw_config.get("ef_construct", 100),
                on_disk=hnsw_config.get("on_disk", False),
            ),
        ),
    )
    console.print(f"[green]  Created collection '{collection_name}' "
                   f"(dim={vector_size}, distance={distance})[/green]")


def setup_text_index(client: QdrantClient, collection_name: str) -> None:
    """Enable BM25 text index for hybrid search."""
    client.create_payload_index(
        collection_name=collection_name,
        field_name="chunk_text",
        field_schema=TextIndexParams(
            type=TextIndexType.TEXT,
            tokenizer=TokenizerType.WORD,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        ),
    )
    console.print("[green]  BM25 text index created on 'chunk_text'[/green]")


def main():
    parser = argparse.ArgumentParser(description="Ingest embeddings into Qdrant for MedLlama")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Config YAML path")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Input embeddings JSONL")
    parser.add_argument("--batch-size", type=int, default=100, help="Upsert batch size")
    parser.add_argument(
        "--local", type=str, default=None,
        help="Use local file-based Qdrant (no Docker). Provide storage directory path.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    qdrant_cfg = config["qdrant"]

    host = qdrant_cfg["host"]
    port = qdrant_cfg["port"]
    collection_name = qdrant_cfg["collection_name"]
    vector_size = qdrant_cfg["vector_size"]
    distance = qdrant_cfg["distance"]
    hnsw_config = qdrant_cfg.get("hnsw", {})

    console.print("[bold blue]MedLlama Qdrant Ingestion[/bold blue]")
    console.print(f"  Collection: {collection_name}")
    console.print(f"  Vector size: {vector_size}")
    console.print(f"  Input: {args.input}")

    if args.local:
        local_path = Path(args.local)
        local_path.mkdir(parents=True, exist_ok=True)
        console.print(f"  Mode: local file-based ({local_path})")
        console.print()
        client = QdrantClient(path=str(local_path))
    else:
        console.print(f"  Host: {host}:{port}")
        console.print()
        # Ensure Qdrant is running
        console.print("[bold]Checking Qdrant...[/bold]")
        if not ensure_qdrant_running(host, port):
            sys.exit(1)
        client = QdrantClient(host=host, port=port, timeout=60)

    # Load embeddings
    console.print("\n[bold]Loading embeddings...[/bold]")
    records = load_embeddings(args.input)
    console.print(f"  Loaded {len(records)} embedded chunks")

    if not records:
        console.print("[red]No records to ingest. Exiting.[/red]")
        sys.exit(1)

    # Detect actual embedding dimension
    actual_dim = len(records[0]["embedding"])
    if actual_dim != vector_size:
        console.print(f"[yellow]  Adjusting vector_size from {vector_size} to {actual_dim}[/yellow]")
        vector_size = actual_dim

    # Create collection
    console.print("\n[bold]Creating collection...[/bold]")
    create_collection(client, collection_name, vector_size, distance, hnsw_config)

    # Setup BM25 text index
    console.print("\n[bold]Setting up BM25 text index...[/bold]")
    setup_text_index(client, collection_name)

    # Upsert points
    console.print(f"\n[bold]Upserting {len(records)} points (batch_size={args.batch_size})...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total_batches = (len(records) + args.batch_size - 1) // args.batch_size
        task = progress.add_task("Upserting...", total=total_batches)

        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            points = []
            for i, record in enumerate(batch):
                point_id = start + i
                payload = {
                    "pmid": record["pmid"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "chunk_text": record["chunk_text"],
                    "chunk_id": record["chunk_id"],
                    "journal": record.get("journal", ""),
                    "pub_date": record.get("pub_date", ""),
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=record["embedding"],
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            progress.advance(task)

    # Verify
    console.print("\n[bold]Verifying ingestion...[/bold]")
    collection_info = client.get_collection(collection_name)
    point_count = collection_info.points_count
    expected = len(records)

    console.print(f"  Expected points: {expected}")
    console.print(f"  Actual points:   {point_count}")

    if point_count == expected:
        console.print(f"[bold green]  Verification passed! All {point_count} points ingested.[/bold green]")
    else:
        console.print(f"[yellow]  Warning: count mismatch ({point_count} != {expected})[/yellow]")

    # Print collection info
    console.print(f"\n[bold]Collection info:[/bold]")
    console.print(f"  Name: {collection_name}")
    console.print(f"  Points: {point_count}")
    console.print(f"  Vectors config: dim={vector_size}, distance={distance}")
    console.print(f"  Status: {collection_info.status}")

    console.print("\n[bold green]Qdrant ingestion complete![/bold green]")


if __name__ == "__main__":
    main()
