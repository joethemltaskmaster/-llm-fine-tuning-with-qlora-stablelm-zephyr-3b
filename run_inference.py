#!/usr/bin/env python
"""
scripts/run_inference.py — quick interactive testing after training.

Loads the base model + fine-tuned adapter once, then drops into a
simple prompt loop for fast manual testing — no flags needed for the
common case. For scripted/one-off calls with full argument control,
use `python -m llm_finetune.inference --prompt "..."` directly instead.

Usage
-----
    python scripts/run_inference.py
    python scripts/run_inference.py --adapter_path ./results/checkpoint-25

Type 'exit' or 'quit' (or press Ctrl+C) to stop.
"""

import argparse
import sys

# Allow running this script directly from scripts/ without installing
# the package first.
sys.path.insert(0, "src")

from llm_finetune.config import default_config
from llm_finetune.inference import load_finetuned_model, stream
from llm_finetune.model import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick interactive test loop for a fine-tuned adapter.")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the saved LoRA adapter directory. Defaults to config.new_model.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=300)
    args = parser.parse_args()

    cfg = default_config
    adapter_path = args.adapter_path or cfg.new_model

    print(f"Loading base model '{cfg.model_name}' with adapter '{adapter_path}'...")
    tokenizer = load_tokenizer(cfg)
    model = load_finetuned_model(adapter_path, cfg)
    print("Ready. Type a prompt (or 'exit' / 'quit' to stop).\n")

    while True:
        try:
            user_prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        stream(user_prompt, model, tokenizer, max_new_tokens=args.max_new_tokens)
        print()  # spacing between turns


if __name__ == "__main__":
    main()
