#!/usr/bin/env python3
"""
MedLlama PubMed Abstract Downloader

Downloads PubMed abstracts using Biopython's Entrez API for use in the
MedLlama RAG pipeline. Searches across multiple medical search terms,
filters for quality, and saves as JSONL.

Usage:
    python src/data_prep/medllama-pubmed-ingest.py [--config CONFIG] [--output OUTPUT] [--max-abstracts N]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from Bio import Entrez, Medline
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
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "pubmed" / "medllama-pubmed-abstracts.jsonl"


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def search_pubmed(term: str, retmax: int, email: str) -> list[str]:
    """Search PubMed for a given term and return list of PMIDs."""
    Entrez.email = email
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=term,
            retmax=retmax,
            sort="relevance",
            usehistory="y",
        )
        results = Entrez.read(handle)
        handle.close()
        return results.get("IdList", [])
    except Exception as e:
        console.print(f"[red]Error searching for '{term}': {e}[/red]")
        return []


def fetch_abstracts_batch(
    pmids: list[str], email: str, batch_size: int = 200, delay: float = 0.5
) -> list[dict]:
    """Fetch abstract details for a list of PMIDs in batches."""
    Entrez.email = email
    all_records = []

    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="medline",
                retmode="text",
            )
            records = list(Medline.parse(handle))
            handle.close()
            all_records.extend(records)
        except Exception as e:
            console.print(f"[yellow]Warning: batch fetch error at offset {start}: {e}[/yellow]")
            # Retry with smaller batch
            for pmid in batch:
                try:
                    time.sleep(delay)
                    handle = Entrez.efetch(
                        db="pubmed",
                        id=pmid,
                        rettype="medline",
                        retmode="text",
                    )
                    records = list(Medline.parse(handle))
                    handle.close()
                    all_records.extend(records)
                except Exception as e2:
                    console.print(f"[yellow]Warning: failed to fetch PMID {pmid}: {e2}[/yellow]")
        time.sleep(delay)

    return all_records


def parse_record(record: dict) -> Optional[dict]:
    """Parse a Medline record into our standard format."""
    pmid = record.get("PMID", "")
    title = record.get("TI", "")
    abstract = record.get("AB", "")
    authors = record.get("AU", [])
    journal = record.get("JT", record.get("TA", ""))
    pub_date = record.get("DP", "")
    language = record.get("LA", [""])

    # Filter: must have abstract
    if not abstract:
        return None

    # Filter: English only
    if isinstance(language, list):
        if language and language[0].lower() not in ("eng", "english", ""):
            return None
    elif isinstance(language, str):
        if language.lower() not in ("eng", "english", ""):
            return None

    # Filter: minimum length
    if len(abstract) < 100:
        return None

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors if isinstance(authors, list) else [authors],
        "journal": journal,
        "pub_date": pub_date,
    }


def main():
    parser = argparse.ArgumentParser(description="Download PubMed abstracts for MedLlama RAG")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to RAG config YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-abstracts",
        type=int,
        default=None,
        help="Override max abstracts to download (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for Entrez efetch calls",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Delay between API calls in seconds",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    pubmed_cfg = config["pubmed"]

    email = pubmed_cfg["email"]
    max_abstracts = args.max_abstracts or pubmed_cfg["max_abstracts"]
    search_terms = pubmed_cfg["search_terms"]
    retmax_per_query = pubmed_cfg["retmax_per_query"]
    min_length = pubmed_cfg.get("min_abstract_length", 100)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]MedLlama PubMed Abstract Downloader[/bold blue]")
    console.print(f"  Email: {email}")
    console.print(f"  Target abstracts: {max_abstracts}")
    console.print(f"  Search terms: {len(search_terms)}")
    console.print(f"  Per-query max: {retmax_per_query}")
    console.print(f"  Output: {output_path}")
    console.print()

    # Phase 1: Search for PMIDs
    all_pmids = []
    seen_pmids = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching PubMed...", total=len(search_terms))

        for term in search_terms:
            progress.update(task, description=f"Searching: {term}")
            pmids = search_pubmed(term, retmax_per_query, email)
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)
            all_pmids.extend(new_pmids)
            console.print(f"  [green]'{term}'[/green]: {len(pmids)} results, {len(new_pmids)} new")
            progress.advance(task)
            time.sleep(args.delay)

    console.print(f"\n[bold]Total unique PMIDs: {len(all_pmids)}[/bold]")

    # Cap at max_abstracts (fetch a bit more to account for filtering)
    fetch_limit = min(len(all_pmids), int(max_abstracts * 1.3))
    pmids_to_fetch = all_pmids[:fetch_limit]
    console.print(f"Fetching details for {len(pmids_to_fetch)} PMIDs...")

    # Phase 2: Fetch abstracts
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total_batches = (len(pmids_to_fetch) + args.batch_size - 1) // args.batch_size
        task = progress.add_task("Fetching abstracts...", total=total_batches)

        all_records = []
        for start in range(0, len(pmids_to_fetch), args.batch_size):
            batch = pmids_to_fetch[start : start + args.batch_size]
            records = fetch_abstracts_batch(batch, email, batch_size=len(batch), delay=args.delay)
            all_records.extend(records)
            progress.advance(task)

    console.print(f"Fetched {len(all_records)} raw records")

    # Phase 3: Parse and filter
    parsed = []
    seen_ids = set()
    for record in all_records:
        result = parse_record(record)
        if result and result["pmid"] not in seen_ids:
            seen_ids.add(result["pmid"])
            parsed.append(result)
            if len(parsed) >= max_abstracts:
                break

    console.print(f"After filtering: {len(parsed)} valid abstracts")

    # Phase 4: Save to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in parsed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    console.print(f"\n[bold green]Saved {len(parsed)} abstracts to {output_path}[/bold green]")

    # Print summary stats
    total_chars = sum(len(r["abstract"]) for r in parsed)
    avg_len = total_chars / len(parsed) if parsed else 0
    console.print(f"  Average abstract length: {avg_len:.0f} chars")
    console.print(f"  Total text: {total_chars:,} chars")
    console.print(f"  Unique journals: {len(set(r['journal'] for r in parsed))}")


if __name__ == "__main__":
    main()
