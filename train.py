"""
train.py — training entry point.

Wires config.py + data.py + model.py into an SFTTrainer run.
Supports CLI overrides of key hyperparameters and resuming from a
saved checkpoint.

Usage
-----
    python -m llm_finetune.train
    python -m llm_finetune.train --max_seq_length 256 --lora_r 16
    python -m llm_finetune.train --resume_from_checkpoint ./results/checkpoint-25
"""

import argparse
import dataclasses
import os

import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from llm_finetune.config import FineTuneConfig, default_config
from llm_finetune.data import load_training_dataset
from llm_finetune.model import load_model_and_tokenizer
from llm_finetune.utils import clear_gpu_memory, set_cuda_alloc_conf


def parse_args() -> argparse.Namespace:
    """
    Parse command-line overrides for FineTuneConfig fields.

    Only a subset of commonly-tuned fields are exposed as flags; any
    FineTuneConfig field can still be set by constructing a config
    directly in code if needed.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with QLoRA.")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--new_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)

    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)

    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory (e.g. ./results/checkpoint-25) to resume from. "
             "Pass 'auto' to resume from the latest checkpoint in output_dir if one exists.",
    )

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> FineTuneConfig:
    """
    Start from default_config and apply any non-None CLI overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments from parse_args().

    Returns
    -------
    FineTuneConfig
    """
    overrides = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(FineTuneConfig)
        if hasattr(args, field.name) and getattr(args, field.name) is not None
    }
    return dataclasses.replace(default_config, **overrides)


def resolve_resume_checkpoint(resume_arg: str, output_dir: str):
    """
    Resolve the --resume_from_checkpoint argument into a value usable
    by Trainer.train(resume_from_checkpoint=...).

    Parameters
    ----------
    resume_arg : str
        Either an explicit checkpoint path, "auto", or None.
    output_dir : str
        Training output directory to search for checkpoints when
        resume_arg == "auto".

    Returns
    -------
    str, bool, or None
        A checkpoint path, True (let Trainer auto-detect the latest
        checkpoint in output_dir), or None (train from scratch).
    """
    if resume_arg is None:
        return None
    if resume_arg == "auto":
        # True tells Trainer to look for the latest checkpoint in output_dir itself.
        if os.path.isdir(output_dir) and any(
            name.startswith("checkpoint-") for name in os.listdir(output_dir)
        ):
            return True
        print(f"No checkpoints found in {output_dir}; starting from scratch.")
        return None
    if not os.path.isdir(resume_arg):
        raise FileNotFoundError(f"Checkpoint path not found: {resume_arg}")
    return resume_arg


def build_training_arguments(config: FineTuneConfig) -> TrainingArguments:
    """
    Construct TrainingArguments from a FineTuneConfig.

    Parameters
    ----------
    config : FineTuneConfig

    Returns
    -------
    TrainingArguments
    """
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        report_to=config.report_to,
    )


def run_training(config: FineTuneConfig, resume_from_checkpoint=None) -> None:
    """
    Execute the full training pipeline: load data, model, tokenizer,
    build the trainer, and run trainer.train().

    Parameters
    ----------
    config : FineTuneConfig
        Fully-resolved configuration (defaults + any CLI overrides).
    resume_from_checkpoint : str, bool, or None
        Passed through to Trainer.train(). See resolve_resume_checkpoint().
    """
    set_cuda_alloc_conf()

    dataset = load_training_dataset(config)
    model, tokenizer, peft_config = load_model_and_tokenizer(config)

    training_arguments = build_training_arguments(config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_arguments,
    )

    clear_gpu_memory()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.model.save_pretrained(config.new_model)
    print(f"Model adapter saved to: {config.new_model}")


def main() -> None:
    """CLI entry point (wired up as `llm-finetune-train` in setup.py)."""
    args = parse_args()
    config = build_config_from_args(args)
    resume = resolve_resume_checkpoint(args.resume_from_checkpoint, config.output_dir)

    run_training(config, resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
