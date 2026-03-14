#!/usr/bin/env python3
"""
MedLlama SFT Training

Full fine-tuning with FSDP across multiple GPUs, or QLoRA fallback on a single GPU.
Uses TRL's SFTTrainer for supervised fine-tuning on medical instruction data.

Usage (FSDP 4x GPU):
    torchrun --nproc_per_node=4 src/training/medllama-sft-train.py --config configs/medllama-training-sft.yaml

Usage (single GPU QLoRA):
    CUDA_VISIBLE_DEVICES=2 python3 src/training/medllama-sft-train.py --config configs/medllama-training-sft.yaml --qlora

Usage (auto-detect):
    python3 src/training/medllama-sft-train.py --config configs/medllama-training-sft.yaml
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

console = Console()

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "medllama-training-sft.yaml"
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sft_data(train_path: str, eval_path: str) -> tuple[Dataset, Dataset]:
    """Load SFT JSONL data into HuggingFace Datasets."""
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

    console.print(f"  Train samples: {len(train_records)}")
    console.print(f"  Eval samples: {len(eval_records)}")

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)

    return train_ds, eval_ds


def setup_qlora_model(model_name: str, qlora_cfg: dict):
    """Load model with QLoRA 4-bit quantization and LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora_cfg.get("quantization", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_modules = qlora_cfg.get("target_modules", "all-linear")
    if target_modules == "all-linear":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=qlora_cfg.get("lora_rank", 64),
        lora_alpha=qlora_cfg.get("lora_alpha", 128),
        lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def setup_full_ft_model(model_name: str):
    """Load model for full fine-tuning (FSDP compatible)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="MedLlama SFT Training")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--qlora", action="store_true", help="Force QLoRA mode")
    parser.add_argument("--local-rank", type=int, default=-1, help="For distributed training")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    wandb_cfg = config.get("wandb", {})
    qlora_cfg = config.get("qlora", {})

    model_name = model_cfg["name"]
    use_qlora = args.qlora or qlora_cfg.get("enabled", False)

    # Detect distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank <= 0

    if is_main:
        console.print("[bold blue]MedLlama SFT Training[/bold blue]")
        console.print(f"  Model: {model_name}")
        console.print(f"  Mode: {'QLoRA' if use_qlora else 'Full Fine-Tuning'}")
        console.print(f"  World size: {world_size}")
        console.print(f"  GPUs: {torch.cuda.device_count()}")
        console.print()

    # Setup W&B
    if wandb_cfg and is_main:
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "medllama")
        os.environ["WANDB_RUN_GROUP"] = "sft"

    # Load tokenizer
    if is_main:
        console.print("[bold]Loading tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load data
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / data_cfg["train_file"]
    eval_path = project_root / data_cfg["eval_file"]

    if is_main:
        console.print("[bold]Loading data...[/bold]")
    train_ds, eval_ds = load_sft_data(str(train_path), str(eval_path))

    # Load model
    if is_main:
        console.print(f"[bold]Loading model ({model_name})...[/bold]")
    if use_qlora:
        model = setup_qlora_model(model_name, qlora_cfg)
        lr = qlora_cfg.get("learning_rate", 2e-4)
        epochs = qlora_cfg.get("num_train_epochs", 3)
        fsdp_config = None
    else:
        model = setup_full_ft_model(model_name)
        lr = train_cfg["learning_rate"]
        epochs = train_cfg["num_train_epochs"]
        # FSDP config for multi-GPU
        if world_size > 1:
            fsdp_config = {
                "fsdp": "full_shard auto_wrap",
                "fsdp_config": {
                    "backward_prefetch": "backward_pre",
                    "forward_prefetch": True,
                    "use_orig_params": True,
                },
            }
        else:
            fsdp_config = None

    # Training arguments
    output_dir = str(project_root / output_cfg["dir"])

    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": lr,
        "weight_decay": train_cfg["weight_decay"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "lr_scheduler_type": train_cfg["lr_scheduler_type"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "bf16": True,
        "logging_steps": train_cfg["logging_steps"],
        "eval_strategy": train_cfg.get("eval_strategy", "steps"),
        "eval_steps": train_cfg.get("eval_steps", 200),
        "save_strategy": "steps",
        "save_steps": train_cfg.get("eval_steps", 200),
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "wandb" if wandb_cfg else "none",
        "run_name": wandb_cfg.get("run_name", "sft-medllama"),
        "seed": train_cfg["seed"],
        "gradient_checkpointing": train_cfg["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_pin_memory": True,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
    }

    if fsdp_config:
        training_args_dict.update(fsdp_config)

    sft_config = SFTConfig(
        **training_args_dict,
        max_length=model_cfg["max_seq_length"],
        packing=False,
    )

    # Initialize trainer
    if is_main:
        console.print("[bold]Initializing SFTTrainer...[/bold]")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # Train
    if is_main:
        console.print("\n[bold green]Starting SFT training...[/bold green]")

    train_result = trainer.train()

    # Save
    if is_main:
        console.print("\n[bold]Saving model...[/bold]")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Log metrics
        metrics = train_result.metrics
        console.print(f"\n[bold]Training metrics:[/bold]")
        for k, v in metrics.items():
            console.print(f"  {k}: {v}")

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Eval
        console.print("\n[bold]Running evaluation...[/bold]")
        eval_metrics = trainer.evaluate()
        console.print(f"\n[bold]Eval metrics:[/bold]")
        for k, v in eval_metrics.items():
            console.print(f"  {k}: {v}")

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        console.print(f"\n[bold green]SFT training complete! Model saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    main()
