#!/usr/bin/env python3
"""
MedLlama Medical QA Evaluation

Evaluates the fine-tuned MedLlama model on MedQA-USMLE-4-options test split.
Compares fine-tuned model accuracy against the base Qwen2.5-7B-Instruct model.

Usage:
    CUDA_VISIBLE_DEVICES=2 python src/eval/medllama-medical-evaluate.py --num-samples 200
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
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
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "merged"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
RESULT_PATH = PROJECT_ROOT / "result" / "eval" / "medllama-medqa-results.json"

SYSTEM_PROMPT = (
    "You are a medical expert taking a multiple-choice exam. "
    "Select the single best answer. Respond with ONLY the letter (A, B, C, or D) "
    "of the correct answer. Do not explain."
)


def format_question(question: str, options: dict) -> str:
    """Format a MedQA question as a multiple-choice prompt."""
    lines = [question, ""]
    for letter in ["A", "B", "C", "D"]:
        if letter in options:
            lines.append(f"{letter}. {options[letter]}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def extract_answer(response: str) -> str:
    """Extract the chosen option letter from model response."""
    response = response.strip()

    # Direct single letter
    if response.upper() in ["A", "B", "C", "D"]:
        return response.upper()

    # Starts with letter followed by punctuation or space
    match = re.match(r"^([A-Da-d])[.\s\):]", response)
    if match:
        return match.group(1).upper()

    # "The answer is X" pattern
    match = re.search(r"(?:the\s+)?answer\s+is\s+([A-Da-d])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for standalone letter in first line
    first_line = response.split("\n")[0].strip()
    match = re.search(r"\b([A-Da-d])\b", first_line)
    if match:
        return match.group(1).upper()

    # Last resort: any A-D in response
    matches = re.findall(r"[A-Da-d]", response)
    if matches:
        return matches[0].upper()

    return ""


def load_model_and_tokenizer(model_path: str, device: str = "cuda:0"):
    """Load model and tokenizer."""
    console.print(f"[bold]Loading model from {model_path}...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    console.print(f"[green]Model loaded on {device}[/green]")
    return model, tokenizer


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    question: str,
    options: dict,
    max_new_tokens: int = 32,
) -> str:
    """Generate answer for a single question."""
    prompt_text = format_question(question, options)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def evaluate_model(
    model,
    tokenizer,
    dataset,
    num_samples: int,
    model_name: str,
) -> dict:
    """Evaluate a model on the MedQA dataset."""
    correct = 0
    total = 0
    results = []

    # Map answer_idx to letter
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Evaluating {model_name}", total=num_samples)

        for i in range(num_samples):
            sample = dataset[i]
            question = sample["question"]
            options = sample["options"]
            answer_idx = sample["answer_idx"]
            if isinstance(answer_idx, int):
                ground_truth = idx_to_letter.get(answer_idx, str(answer_idx))
            else:
                ground_truth = str(answer_idx).strip()

            # If ground_truth is already a letter (dataset format varies)
            if ground_truth not in ["A", "B", "C", "D"]:
                # Try direct string
                gt_str = str(sample.get("answer_idx", ""))
                if gt_str in ["A", "B", "C", "D"]:
                    ground_truth = gt_str
                else:
                    ground_truth = idx_to_letter.get(int(answer_idx), "A")

            response = generate_answer(model, tokenizer, question, options)
            predicted = extract_answer(response)
            is_correct = predicted == ground_truth

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "index": i,
                "question": question[:200],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "raw_response": response[:200],
                "correct": is_correct,
            })

            progress.advance(task)

    accuracy = correct / total if total > 0 else 0.0
    return {
        "model_name": model_name,
        "num_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="MedLlama MedQA Evaluation")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of test samples")
    parser.add_argument("--merged-model", type=str, default=str(MERGED_MODEL_PATH), help="Path to merged model")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_NAME, help="Base model name")
    parser.add_argument("--output", type=str, default=str(RESULT_PATH), help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for model")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model evaluation")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "offline"

    console.print("[bold blue]MedLlama MedQA Evaluation[/bold blue]")
    console.print(f"  Samples: {args.num_samples}")
    console.print(f"  Merged model: {args.merged_model}")
    console.print(f"  Base model: {args.base_model}")
    console.print(f"  Device: {args.device}")
    console.print()

    # Load dataset
    console.print("[bold]Loading MedQA-USMLE-4-options test split...[/bold]")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    console.print(f"  Total test samples: {len(dataset)}")

    num_samples = min(args.num_samples, len(dataset))
    console.print(f"  Using {num_samples} samples for evaluation\n")

    start_time = time.time()

    # Evaluate fine-tuned model
    console.print("[bold cyan]--- Fine-tuned Model Evaluation ---[/bold cyan]")
    ft_model, ft_tokenizer = load_model_and_tokenizer(args.merged_model, args.device)
    ft_results = evaluate_model(ft_model, ft_tokenizer, dataset, num_samples, "medllama-finetuned")

    # Free memory
    del ft_model
    torch.cuda.empty_cache()
    console.print(f"[green]Fine-tuned accuracy: {ft_results['accuracy']:.4f} ({ft_results['correct']}/{ft_results['num_samples']})[/green]\n")

    # Evaluate base model
    base_results = None
    if not args.skip_base:
        console.print("[bold cyan]--- Base Model Evaluation ---[/bold cyan]")
        base_model, base_tokenizer = load_model_and_tokenizer(args.base_model, args.device)
        base_results = evaluate_model(base_model, base_tokenizer, dataset, num_samples, "qwen2.5-7b-instruct-base")
        del base_model
        torch.cuda.empty_cache()
        console.print(f"[green]Base accuracy: {base_results['accuracy']:.4f} ({base_results['correct']}/{base_results['num_samples']})[/green]\n")

    total_time = time.time() - start_time

    # Build final output
    output = {
        "eval_type": "medqa_accuracy",
        "timestamp": datetime.now().isoformat(),
        "dataset": "GBaker/MedQA-USMLE-4-options",
        "num_samples": num_samples,
        "total_time_seconds": round(total_time, 1),
        "finetuned": {
            "model": args.merged_model,
            "accuracy": ft_results["accuracy"],
            "correct": ft_results["correct"],
            "total": ft_results["num_samples"],
        },
    }

    if base_results:
        output["base"] = {
            "model": args.base_model,
            "accuracy": base_results["accuracy"],
            "correct": base_results["correct"],
            "total": base_results["num_samples"],
        }
        improvement = ft_results["accuracy"] - base_results["accuracy"]
        output["improvement"] = round(improvement, 4)

    # Include per-sample results (truncated questions)
    output["finetuned_details"] = ft_results["results"]
    if base_results:
        output["base_details"] = base_results["results"]

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    console.print(f"[bold green]Results saved to {output_path}[/bold green]")

    # Print summary table
    table = Table(title="MedQA Evaluation Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Correct / Total")
    table.add_row(
        "MedLlama (fine-tuned)",
        f"{ft_results['accuracy']:.4f}",
        f"{ft_results['correct']}/{ft_results['num_samples']}",
    )
    if base_results:
        table.add_row(
            "Qwen2.5-7B-Instruct (base)",
            f"{base_results['accuracy']:.4f}",
            f"{base_results['correct']}/{base_results['num_samples']}",
        )
    console.print(table)

    # W&B logging
    try:
        import wandb
        wandb.init(
            project="medllama-eval",
            name=f"medqa-eval-{num_samples}",
            config={"num_samples": num_samples, "dataset": "MedQA-USMLE-4-options"},
        )
        wandb.log({
            "finetuned_accuracy": ft_results["accuracy"],
            "finetuned_correct": ft_results["correct"],
        })
        if base_results:
            wandb.log({
                "base_accuracy": base_results["accuracy"],
                "base_correct": base_results["correct"],
                "improvement": improvement,
            })
        wandb.finish()
        console.print("[green]Logged to W&B (offline)[/green]")
    except Exception as e:
        console.print(f"[yellow]W&B logging failed: {e}[/yellow]")


if __name__ == "__main__":
    main()
