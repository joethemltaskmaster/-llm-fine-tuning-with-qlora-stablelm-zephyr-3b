"""
inference.py — streaming text generation with a fine-tuned LoRA adapter.

Loads the quantized base model, applies the saved LoRA adapter via
PeftModel, builds the instruction prompt, and streams the generated
response token-by-token.

Usage
-----
    python -m llm_finetune.inference --prompt "Tell me about operations research."
    python -m llm_finetune.inference --prompt "..." --adapter_path ./llama-2-7b-mini-ult-guanaco --max_new_tokens 300
"""

import argparse
from typing import Optional

from peft import PeftModel
from transformers import TextStreamer

from llm_finetune.config import FineTuneConfig, default_config
from llm_finetune.data import format_instruction_prompt
from llm_finetune.model import load_model, load_tokenizer


def load_finetuned_model(
    adapter_path: str,
    config: Optional[FineTuneConfig] = None,
):
    """
    Load the quantized base model and apply a saved LoRA adapter.

    Parameters
    ----------
    adapter_path : str
        Path to the saved adapter directory (trainer.model.save_pretrained
        output from train.py, i.e. config.new_model).
    config : FineTuneConfig, optional
        Defaults to `default_config` if not provided. Determines the
        base model and quantization settings the adapter was trained
        against.

    Returns
    -------
    PeftModel
        Base model wrapped with the LoRA adapter, ready for generation.
    """
    cfg = config or default_config

    base_model = load_model(cfg)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model


def stream(
    user_prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 500,
) -> None:
    """
    Generate and stream a response to `user_prompt` token-by-token.

    Parameters
    ----------
    user_prompt : str
        The instruction to send the model.
    model
        A loaded (adapter-applied) causal LM on a CUDA device.
    tokenizer
        The corresponding tokenizer.
    max_new_tokens : int
        Maximum number of tokens to generate.
    """
    runtime_flag = "cuda:0"
    prompt = format_instruction_prompt(user_prompt)

    inputs = tokenizer([prompt], return_tensors="pt").to(runtime_flag)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference."""
    parser = argparse.ArgumentParser(description="Run streaming inference with a fine-tuned adapter.")

    parser.add_argument("--prompt", type=str, required=True, help="Instruction/prompt to send the model.")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the saved LoRA adapter directory. Defaults to config.new_model.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=500)

    return parser.parse_args()


def main() -> None:
    """CLI entry point (wired up as `llm-finetune-infer` in setup.py)."""
    args = parse_args()
    cfg = default_config
    adapter_path = args.adapter_path or cfg.new_model

    tokenizer = load_tokenizer(cfg)
    model = load_finetuned_model(adapter_path, cfg)

    stream(args.prompt, model, tokenizer, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
