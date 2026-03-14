#!/usr/bin/env python3
"""
MedLlama DPO Training

Direct Preference Optimization training using the SFT checkpoint as base model
and the original model as reference. Uses TRL's DPOTrainer.

Usage (FSDP 4x GPU):
    torchrun --nproc_per_node=4 src/training/medllama-dpo-train.py --config configs/medllama-training-dpo.yaml

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=2 python3 src/training/medllama-dpo-train.py --config configs/medllama-training-dpo.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

console = Console()

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "medllama-training-dpo.yaml"
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dpo_data(train_path: str, eval_path: str) -> tuple[Dataset, Dataset]:
    """Load DPO JSONL data into HuggingFace Datasets."""
    def load_jsonl(path: str) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    train_records = load_jsonl(train_path)
    eval_records = load_jsonl(eval_path)

    console.print(f"  Train pairs: {len(train_records)}")
    console.print(f"  Eval pairs: {len(eval_records)}")

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)

    return train_ds, eval_ds


def main():
    parser = argparse.ArgumentParser(description="MedLlama DPO Training")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--local-rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    wandb_cfg = config.get("wandb", {})

    model_name = model_cfg["name"]
    ref_model_name = model_cfg.get("ref_model", model_name)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank <= 0

    if is_main:
        console.print("[bold blue]MedLlama DPO Training[/bold blue]")
        console.print(f"  Policy model: {model_name}")
        console.print(f"  Reference model: {ref_model_name}")
        console.print(f"  Beta: {train_cfg['beta']}")
        console.print(f"  World size: {world_size}")
        console.print()

    # Setup W&B
    if wandb_cfg and is_main:
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "medllama")
        os.environ["WANDB_RUN_GROUP"] = "dpo"

    # Load tokenizer
    if is_main:
        console.print("[bold]Loading tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO requires left padding

    # Load data
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / data_cfg["train_file"]
    eval_path = project_root / data_cfg["eval_file"]

    if is_main:
        console.print("[bold]Loading data...[/bold]")
    train_ds, eval_ds = load_dpo_data(str(train_path), str(eval_path))

    # Load policy model (SFT checkpoint)
    if is_main:
        console.print(f"[bold]Loading policy model ({model_name})...[/bold]")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Load reference model
    if is_main:
        console.print(f"[bold]Loading reference model ({ref_model_name})...[/bold]")

    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # Training arguments
    output_dir = str(project_root / output_cfg["dir"])

    # FSDP config for multi-GPU
    fsdp_kwargs = {}
    if world_size > 1:
        fsdp_kwargs = {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "backward_prefetch": "backward_pre",
                "forward_prefetch": True,
                "use_orig_params": True,
            },
        }

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=train_cfg["beta"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=True,
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=2,
        report_to="wandb" if wandb_cfg else "none",
        run_name=wandb_cfg.get("run_name", "dpo-medllama"),
        seed=train_cfg["seed"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=model_cfg["max_seq_length"],
        max_prompt_length=model_cfg["max_seq_length"] // 2,
        remove_unused_columns=False,
        **fsdp_kwargs,
    )

    # Initialize DPO trainer
    if is_main:
        console.print("[bold]Initializing DPOTrainer...[/bold]")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # Train
    if is_main:
        console.print("\n[bold green]Starting DPO training...[/bold green]")

    train_result = trainer.train()

    # Save
    if is_main:
        console.print("\n[bold]Saving model...[/bold]")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        metrics = train_result.metrics
        console.print(f"\n[bold]Training metrics:[/bold]")
        for k, v in metrics.items():
            console.print(f"  {k}: {v}")

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        console.print(f"\n[bold green]DPO training complete! Model saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    main()
