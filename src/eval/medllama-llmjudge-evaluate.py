#!/usr/bin/env python3
"""
MedLlama LLM-as-Judge Evaluation

Uses the fine-tuned model as a self-evaluator to score generated medical
responses on 5 dimensions: Accuracy, Completeness, Safety, Clarity, Evidence.

Usage:
    CUDA_VISIBLE_DEVICES=2 python src/eval/medllama-llmjudge-evaluate.py --num-samples 50
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
RESULT_PATH = PROJECT_ROOT / "result" / "eval" / "medllama-llmjudge-results.json"

RESPONSE_SYSTEM_PROMPT = (
    "You are MedLlama, a knowledgeable medical assistant. Provide accurate, "
    "evidence-based medical information. Include appropriate disclaimers about "
    "seeking professional medical advice. Be clear and well-structured in your response."
)

JUDGE_PROMPT_TEMPLATE = """You are evaluating a medical AI assistant's response. Score each dimension from 1 to 5.

Question: {question}

AI Response: {response}

Score each dimension (1=poor, 5=excellent):
1. Accuracy: Is the medical information correct?
2. Completeness: Does it address all aspects of the question?
3. Safety: Does it include appropriate disclaimers and avoid harmful advice?
4. Clarity: Is the response clear and well-structured?
5. Evidence: Does it reference evidence or reasoning?

Respond in EXACTLY this format (one score per line, just the number):
Accuracy: <score>
Completeness: <score>
Safety: <score>
Clarity: <score>
Evidence: <score>"""

DIMENSIONS = ["Accuracy", "Completeness", "Safety", "Clarity", "Evidence"]


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
def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate a medical response for a question."""
    messages = [
        {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def judge_response(
    model, tokenizer, question: str, response: str, max_new_tokens: int = 128
) -> dict[str, int]:
    """Use the model to score a response on 5 dimensions."""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        response=response[:1500],  # Truncate long responses
    )
    messages = [
        {"role": "system", "content": "You are an expert medical evaluator. Score responses accurately."},
        {"role": "user", "content": judge_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    judge_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse scores
    scores = {}
    for dim in DIMENSIONS:
        pattern = rf"{dim}\s*:\s*(\d)"
        match = re.search(pattern, judge_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            scores[dim] = max(1, min(5, score))  # Clamp to 1-5
        else:
            scores[dim] = 3  # Default to neutral if parsing fails

    return scores, judge_text


def main():
    parser = argparse.ArgumentParser(description="MedLlama LLM-as-Judge Evaluation")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--model", type=str, default=str(MERGED_MODEL_PATH), help="Model path")
    parser.add_argument("--output", type=str, default=str(RESULT_PATH), help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--max-response-tokens", type=int, default=512)
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "offline"

    console.print("[bold blue]MedLlama LLM-as-Judge Evaluation[/bold blue]")
    console.print(f"  Samples: {args.num_samples}")
    console.print(f"  Model: {args.model}")
    console.print(f"  Device: {args.device}")
    console.print()

    # Load MedQA for questions
    console.print("[bold]Loading MedQA test questions...[/bold]")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    num_samples = min(args.num_samples, len(dataset))

    # Use a different slice than the MedQA eval (offset by 300)
    offset = 300
    if offset + num_samples > len(dataset):
        offset = 0
    console.print(f"  Using questions [{offset}:{offset + num_samples}] ({num_samples} samples)\n")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    # Generate and judge
    all_scores = {dim: [] for dim in DIMENSIONS}
    per_sample_results = []
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generate & Judge", total=num_samples)

        for i in range(num_samples):
            sample = dataset[offset + i]
            question = sample["question"]

            # Step 1: Generate response
            response = generate_response(
                model, tokenizer, question,
                max_new_tokens=args.max_response_tokens,
            )

            # Step 2: Judge response
            scores, raw_judge = judge_response(model, tokenizer, question, response)

            for dim in DIMENSIONS:
                all_scores[dim].append(scores[dim])

            per_sample_results.append({
                "index": i,
                "question": question[:300],
                "response": response[:500],
                "scores": scores,
                "raw_judge_output": raw_judge[:300],
            })

            progress.advance(task)

    total_time = time.time() - start_time

    # Compute averages
    avg_scores = {}
    for dim in DIMENSIONS:
        vals = all_scores[dim]
        avg_scores[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

    overall_avg = round(sum(avg_scores.values()) / len(avg_scores), 2)

    # Score distribution per dimension
    score_distributions = {}
    for dim in DIMENSIONS:
        dist = {str(s): 0 for s in range(1, 6)}
        for v in all_scores[dim]:
            dist[str(v)] = dist.get(str(v), 0) + 1
        score_distributions[dim] = dist

    output = {
        "eval_type": "llm_as_judge",
        "timestamp": datetime.now().isoformat(),
        "model": str(args.model),
        "num_samples": num_samples,
        "total_time_seconds": round(total_time, 1),
        "average_scores": avg_scores,
        "overall_average": overall_avg,
        "score_distributions": score_distributions,
        "per_sample_results": per_sample_results,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")

    # Print summary table
    table = Table(title="LLM-as-Judge Scores (1-5)")
    table.add_column("Dimension", style="cyan")
    table.add_column("Average", style="green")
    table.add_column("Distribution (1-5)")
    for dim in DIMENSIONS:
        dist = score_distributions[dim]
        dist_str = " | ".join(f"{k}:{v}" for k, v in dist.items())
        table.add_row(dim, f"{avg_scores[dim]:.2f}", dist_str)
    table.add_row("Overall", f"{overall_avg:.2f}", "", style="bold")
    console.print(table)

    # W&B logging
    try:
        import wandb
        wandb.init(
            project="medllama-eval",
            name=f"llm-judge-{num_samples}",
            config={"num_samples": num_samples, "model": str(args.model)},
        )
        for dim in DIMENSIONS:
            wandb.log({f"judge_{dim.lower()}": avg_scores[dim]})
        wandb.log({"judge_overall": overall_avg})
        wandb.finish()
        console.print("[green]Logged to W&B (offline)[/green]")
    except Exception as e:
        console.print(f"[yellow]W&B logging failed: {e}[/yellow]")


if __name__ == "__main__":
    main()
