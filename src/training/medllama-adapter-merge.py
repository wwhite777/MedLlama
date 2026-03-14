#!/usr/bin/env python3
"""
MedLlama Adapter Merge

Merges LoRA adapters (SFT and/or DPO) into the base model and saves
the full merged model for deployment with vLLM.

Usage:
    python3 src/training/medllama-adapter-merge.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --adapter checkpoints/dpo \
        --output checkpoints/merged
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="checkpoints/dpo")
    parser.add_argument("--output", type=str, default="checkpoints/merged")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    adapter_path = project_root / args.adapter
    output_path = project_root / args.output
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]MedLlama Adapter Merge[/bold blue]")
    console.print(f"  Base model: {args.base}")
    console.print(f"  Adapter: {adapter_path}")
    console.print(f"  Output: {output_path}")

    # Load base model
    console.print("\n[bold]Loading base model...[/bold]")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load and merge adapter
    console.print("[bold]Loading adapter...[/bold]")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    console.print("[bold]Merging adapter into base model...[/bold]")
    model = model.merge_and_unload()

    # Save merged model
    console.print(f"[bold]Saving merged model to {output_path}...[/bold]")
    model.save_pretrained(str(output_path), safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))

    # Verify
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    console.print(f"\n[bold green]Merge complete![/bold green]")
    console.print(f"  Files: {len(list(output_path.iterdir()))}")
    console.print(f"  Size: {total_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
