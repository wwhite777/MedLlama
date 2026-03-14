"""
MedLlama DPO Data Formatter

Generates ~3K DPO preference pairs from the MedQA dataset.
Chosen = correct answer with explanation.
Rejected = random wrong answer with plausible-sounding but incorrect explanation.

Usage:
    python src/data_prep/medllama-dpo-format.py
    python src/data_prep/medllama-dpo-format.py --output-dir data/dpo --max-samples 3000
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()

WRONG_EXPLANATION_TEMPLATES = [
    (
        "The correct answer is {letter}) {answer}.\n\n"
        "Explanation: In this clinical scenario, {answer} is the most likely answer "
        "because it aligns with the typical presentation described. The pathophysiology "
        "involves mechanisms that are commonly associated with this condition, making "
        "the other options less probable."
    ),
    (
        "The answer is {letter}) {answer}.\n\n"
        "Explanation: Based on the clinical findings presented, {answer} is the "
        "best choice. This is supported by the patient's symptoms and clinical "
        "markers, which are characteristic of this diagnosis. The remaining options "
        "do not adequately explain the full clinical picture."
    ),
    (
        "The correct answer is {letter}) {answer}.\n\n"
        "Explanation: {answer} is the preferred answer here because the clinical "
        "presentation is most consistent with this option. Key features in the "
        "question stem point toward this diagnosis, while the distractors represent "
        "conditions with different expected findings."
    ),
    (
        "I would select {letter}) {answer}.\n\n"
        "Explanation: The scenario described is best explained by {answer}. "
        "The relevant clinical features and laboratory findings, when considered "
        "together, are most consistent with this choice. Other options may share "
        "some similarities but fail to account for the complete presentation."
    ),
    (
        "The correct answer is {letter}) {answer}.\n\n"
        "Explanation: Considering the patient history and examination findings, "
        "{answer} represents the most appropriate response. This is because the "
        "underlying mechanism is directly related to the clinical findings "
        "described in the question."
    ),
]


def generate_dpo_pairs(max_samples: int, seed: int) -> list[dict]:
    """Generate DPO preference pairs from MedQA."""
    console.print("[bold blue]Loading MedQA-USMLE-4-options for DPO...[/]")
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", trust_remote_code=True)
    except Exception as e:
        console.print(f"[red]Failed to load MedQA: {e}[/]")
        console.print("[yellow]Retrying...[/]")
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", trust_remote_code=True)

    random.seed(seed)
    pairs = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices:
        if len(pairs) >= max_samples:
            break
        row = ds[idx]
        question = row.get("question", "")
        options = row.get("options", {})
        answer = row.get("answer", "")

        if not question or not options or not answer:
            continue

        # Build prompt with options
        option_lines = []
        for key in sorted(options.keys()):
            option_lines.append(f"  {key}) {options[key]}")
        options_text = "\n".join(option_lines)

        prompt = (
            f"The following is a medical licensing exam question. "
            f"Select the correct answer and provide a detailed explanation.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}"
        )

        # Find correct letter and wrong options
        correct_letter = None
        wrong_options = []
        for key, val in sorted(options.items()):
            if val == answer:
                correct_letter = key
            else:
                wrong_options.append((key, val))

        if not correct_letter or not wrong_options:
            continue

        # Chosen: correct answer
        chosen = (
            f"The correct answer is {correct_letter}) {answer}.\n\n"
            f"Explanation: This question tests knowledge of clinical medicine. "
            f"The answer is {answer} because it is the most appropriate choice "
            f"based on the clinical scenario and established medical guidelines. "
            f"The other options can be ruled out through careful clinical reasoning."
        )

        # Rejected: pick a random wrong answer with a plausible template
        wrong_letter, wrong_answer = random.choice(wrong_options)
        template = random.choice(WRONG_EXPLANATION_TEMPLATES)
        rejected = template.format(letter=wrong_letter, answer=wrong_answer)

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    console.print(f"  [green]Generated {len(pairs)} DPO preference pairs[/]")
    return pairs


def write_jsonl(data: list[dict], path: Path) -> None:
    """Write data to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    console.print(f"  [green]Wrote {len(data)} entries to {path}[/]")


def main():
    parser = argparse.ArgumentParser(description="Format DPO preference data for MedLlama")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dpo",
        help="Output directory for JSONL files (default: data/dpo)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Maximum DPO pairs to generate (default: 3000)",
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

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = project_root / args.output_dir

    console.rule("[bold]MedLlama DPO Data Formatter[/]")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Max samples: {args.max_samples}")
    console.print(f"Eval ratio: {args.eval_ratio}")
    console.print(f"Seed: {args.seed}")
    console.print()

    # Generate pairs
    pairs = generate_dpo_pairs(args.max_samples, args.seed)

    # Shuffle
    random.seed(args.seed)
    random.shuffle(pairs)

    # Split
    eval_size = max(1, int(len(pairs) * args.eval_ratio))
    train_pairs = pairs[eval_size:]
    eval_pairs = pairs[:eval_size]

    console.print()
    console.print(f"[bold]Train: {len(train_pairs)}, Eval: {len(eval_pairs)}[/]")

    # Stats
    table = Table(title="DPO Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total pairs", str(len(pairs)))
    table.add_row("Train pairs", str(len(train_pairs)))
    table.add_row("Eval pairs", str(len(eval_pairs)))
    if pairs:
        avg_prompt = sum(len(p["prompt"]) for p in pairs) / len(pairs)
        avg_chosen = sum(len(p["chosen"]) for p in pairs) / len(pairs)
        avg_rejected = sum(len(p["rejected"]) for p in pairs) / len(pairs)
        table.add_row("Avg prompt length (chars)", f"{avg_prompt:.0f}")
        table.add_row("Avg chosen length (chars)", f"{avg_chosen:.0f}")
        table.add_row("Avg rejected length (chars)", f"{avg_rejected:.0f}")
    console.print(table)

    # Write
    console.print()
    console.print("[bold]Writing output files...[/]")
    write_jsonl(train_pairs, output_dir / "medllama-dpo-train.jsonl")
    write_jsonl(eval_pairs, output_dir / "medllama-dpo-eval.jsonl")

    console.rule("[bold green]Done[/]")


if __name__ == "__main__":
    main()
