"""
MedLlama SFT Data Formatter

Downloads and formats ~15K SFT examples from MedQA, PubMedQA, and ChatDoctor
into a unified chat-style JSONL format for supervised fine-tuning.

Usage:
    python src/data_prep/medllama-sft-format.py
    python src/data_prep/medllama-sft-format.py --output-dir data/sft --max-per-source 5000
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()

SYSTEM_PROMPT = (
    "You are MedLlama, a knowledgeable medical assistant trained to provide "
    "accurate, evidence-based medical information. You help healthcare "
    "professionals and students understand medical concepts, interpret clinical "
    "scenarios, and reason through diagnostic and treatment decisions. "
    "Always clarify that your responses are for educational purposes and "
    "should not replace professional medical judgment."
)

MIN_INSTRUCTION_LENGTH = 20
MIN_RESPONSE_LENGTH = 30


def make_message(instruction: str, response: str) -> dict:
    """Create a chat-formatted message dict."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": response.strip()},
        ]
    }


def format_medqa(max_samples: int) -> list[dict]:
    """Download and format MedQA-USMLE-4-options from HuggingFace."""
    console.print("[bold blue]Downloading MedQA-USMLE-4-options...[/]")
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", trust_remote_code=True)
    except Exception as e:
        console.print(f"[red]Failed to load MedQA: {e}[/]")
        console.print("[yellow]Retrying...[/]")
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", trust_remote_code=True)

    examples = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices:
        if len(examples) >= max_samples:
            break
        row = ds[idx]
        question = row.get("question", "")
        options = row.get("options", {})
        answer = row.get("answer", "")

        if not question or not options or not answer:
            continue

        # Build instruction with options
        option_lines = []
        for key in sorted(options.keys()):
            option_lines.append(f"  {key}) {options[key]}")
        options_text = "\n".join(option_lines)

        instruction = (
            f"The following is a medical licensing exam question. "
            f"Select the correct answer and provide a detailed explanation.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}"
        )

        # Find the correct option letter
        correct_letter = None
        for key, val in options.items():
            if val == answer:
                correct_letter = key
                break

        if correct_letter:
            response = (
                f"The correct answer is {correct_letter}) {answer}.\n\n"
                f"Explanation: This question tests knowledge of clinical medicine. "
                f"The answer is {answer} because it is the most appropriate choice "
                f"based on the clinical scenario and established medical guidelines. "
                f"The other options can be ruled out through careful clinical reasoning."
            )
        else:
            response = (
                f"The correct answer is {answer}.\n\n"
                f"Explanation: Based on the clinical scenario presented, {answer} "
                f"is the most appropriate answer according to established medical "
                f"knowledge and clinical guidelines."
            )

        if len(instruction) < MIN_INSTRUCTION_LENGTH or len(response) < MIN_RESPONSE_LENGTH:
            continue

        examples.append(make_message(instruction, response))

    console.print(f"  [green]Formatted {len(examples)} MedQA examples[/]")
    return examples


def format_pubmedqa(max_samples: int) -> list[dict]:
    """Download and format PubMedQA (labeled subset) from HuggingFace."""
    console.print("[bold blue]Downloading PubMedQA (pqa_labeled)...[/]")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)
    except Exception as e:
        console.print(f"[red]Failed to load PubMedQA: {e}[/]")
        console.print("[yellow]Retrying...[/]")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)

    examples = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices:
        if len(examples) >= max_samples:
            break
        row = ds[idx]
        question = row.get("question", "")
        context = row.get("context", {})
        long_answer = row.get("long_answer", "")
        final_decision = row.get("final_decision", "")

        if not question or not long_answer:
            continue

        # Build context string from the context dict
        context_text = ""
        if isinstance(context, dict):
            contexts = context.get("contexts", [])
            if isinstance(contexts, list):
                context_text = " ".join(contexts)
        elif isinstance(context, str):
            context_text = context

        if context_text:
            instruction = (
                f"Based on the following biomedical research context, answer the question.\n\n"
                f"Context: {context_text}\n\n"
                f"Question: {question}"
            )
        else:
            instruction = (
                f"Answer the following biomedical research question.\n\n"
                f"Question: {question}"
            )

        decision_str = ""
        if final_decision:
            decision_str = f"\n\nFinal answer: {final_decision}"

        response = f"{long_answer}{decision_str}"

        if len(instruction) < MIN_INSTRUCTION_LENGTH or len(response) < MIN_RESPONSE_LENGTH:
            continue

        examples.append(make_message(instruction, response))

    console.print(f"  [green]Formatted {len(examples)} PubMedQA examples[/]")
    return examples


def format_chatdoctor(max_samples: int) -> list[dict]:
    """Download and format ChatDoctor-HealthCareMagic from HuggingFace."""
    console.print("[bold blue]Downloading ChatDoctor-HealthCareMagic-100k...[/]")
    try:
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train", trust_remote_code=True)
    except Exception as e:
        console.print(f"[red]Failed to load ChatDoctor: {e}[/]")
        console.print("[yellow]Retrying...[/]")
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train", trust_remote_code=True)

    examples = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices:
        if len(examples) >= max_samples:
            break
        row = ds[idx]
        instruction = row.get("input", "")
        response = row.get("output", "")

        if not instruction or not response:
            continue
        if len(instruction) < MIN_INSTRUCTION_LENGTH or len(response) < MIN_RESPONSE_LENGTH:
            continue

        examples.append(make_message(instruction, response))

    console.print(f"  [green]Formatted {len(examples)} ChatDoctor examples[/]")
    return examples


def deduplicate(examples: list[dict]) -> list[dict]:
    """Remove duplicates based on user message content hash."""
    seen = set()
    unique = []
    for ex in examples:
        user_msg = ex["messages"][1]["content"]
        h = hashlib.md5(user_msg.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    removed = len(examples) - len(unique)
    if removed > 0:
        console.print(f"  [yellow]Removed {removed} duplicates[/]")
    return unique


def compute_stats(examples: list[dict], label: str) -> None:
    """Print statistics about the dataset."""
    if not examples:
        console.print(f"[red]No examples in {label}[/]")
        return

    instruction_lengths = []
    response_lengths = []
    for ex in examples:
        instruction_lengths.append(len(ex["messages"][1]["content"]))
        response_lengths.append(len(ex["messages"][2]["content"]))

    table = Table(title=f"{label} Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total examples", str(len(examples)))
    table.add_row("Avg instruction length (chars)", f"{sum(instruction_lengths) / len(instruction_lengths):.0f}")
    table.add_row("Avg response length (chars)", f"{sum(response_lengths) / len(response_lengths):.0f}")
    table.add_row("Min instruction length", str(min(instruction_lengths)))
    table.add_row("Max instruction length", str(max(instruction_lengths)))
    table.add_row("Min response length", str(min(response_lengths)))
    table.add_row("Max response length", str(max(response_lengths)))
    console.print(table)


def write_jsonl(examples: list[dict], path: Path) -> None:
    """Write examples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    console.print(f"  [green]Wrote {len(examples)} examples to {path}[/]")


def main():
    parser = argparse.ArgumentParser(description="Format SFT training data for MedLlama")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sft",
        help="Output directory for JSONL files (default: data/sft)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=5000,
        help="Maximum examples per data source (default: 5000)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.05,
        help="Fraction of data for eval split (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve output dir relative to script location or cwd
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = project_root / args.output_dir

    console.rule("[bold]MedLlama SFT Data Formatter[/]")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Max per source: {args.max_per_source}")
    console.print(f"Eval ratio: {args.eval_ratio}")
    console.print(f"Seed: {args.seed}")
    console.print()

    # Download and format each source
    all_examples = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing data sources...", total=3)

        medqa = format_medqa(args.max_per_source)
        all_examples.extend(medqa)
        progress.advance(task)

        pubmedqa = format_pubmedqa(args.max_per_source)
        all_examples.extend(pubmedqa)
        progress.advance(task)

        chatdoctor = format_chatdoctor(args.max_per_source)
        all_examples.extend(chatdoctor)
        progress.advance(task)

    console.print()

    # Deduplicate
    console.print("[bold]Deduplicating...[/]")
    all_examples = deduplicate(all_examples)

    # Shuffle
    random.shuffle(all_examples)

    # Split
    eval_size = max(1, int(len(all_examples) * args.eval_ratio))
    train_examples = all_examples[eval_size:]
    eval_examples = all_examples[:eval_size]

    console.print()
    console.print(f"[bold]Train: {len(train_examples)}, Eval: {len(eval_examples)}[/]")

    # Per-source counts
    source_counts = {"MedQA": 0, "PubMedQA": 0, "ChatDoctor": 0}
    for ex in all_examples:
        user_content = ex["messages"][1]["content"]
        if "medical licensing exam" in user_content:
            source_counts["MedQA"] += 1
        elif "biomedical research" in user_content:
            source_counts["PubMedQA"] += 1
        else:
            source_counts["ChatDoctor"] += 1

    table = Table(title="Per-Source Counts")
    table.add_column("Source", style="cyan")
    table.add_column("Count", style="green")
    for source, count in source_counts.items():
        table.add_row(source, str(count))
    table.add_row("[bold]Total[/]", f"[bold]{sum(source_counts.values())}[/]")
    console.print(table)

    # Stats
    compute_stats(train_examples, "Train")
    compute_stats(eval_examples, "Eval")

    # Write
    console.print()
    console.print("[bold]Writing output files...[/]")
    write_jsonl(train_examples, output_dir / "medllama-sft-train.jsonl")
    write_jsonl(eval_examples, output_dir / "medllama-sft-eval.jsonl")

    console.rule("[bold green]Done[/]")


if __name__ == "__main__":
    main()
