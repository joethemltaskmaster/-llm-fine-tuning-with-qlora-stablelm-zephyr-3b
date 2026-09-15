"""
llm_finetune
------------

QLoRA fine-tuning of stabilityai/stablelm-zephyr-3b on the
mlabonne/guanaco-llama2-1k instruction dataset.

This package exposes the core building blocks used by the training
and inference entry points:

- config   : hyperparameters and run configuration
- data     : dataset loading/preprocessing
- model    : quantized model, tokenizer, and LoRA setup
- train    : training loop (SFTTrainer)
- inference: streaming text generation with a fine-tuned adapter
- utils    : shared helpers (GPU memory cleanup, logging, etc.)
"""

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "model",
    "train",
    "inference",
    "utils",
]
