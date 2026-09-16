"""
config.py — central configuration for the fine-tuning pipeline.

All hyperparameters and run settings live here instead of being
scattered across scripts. Import `default_config` for the values
used in the original notebook, or construct a `FineTuneConfig`
with overrides for experimentation.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FineTuneConfig:
    # ------------------------------------------------------------------
    # Model / dataset
    # ------------------------------------------------------------------
    model_name: str = "stabilityai/stablelm-zephyr-3b"
    dataset_name: str = "mlabonne/guanaco-llama2-1k"
    new_model: str = "llama-2-7b-mini-ult-guanaco"

    # ------------------------------------------------------------------
    # LoRA configuration
    # ------------------------------------------------------------------
    lora_r: int = 8              # LoRA attention dimension
    lora_alpha: int = 16         # LoRA scaling alpha
    lora_dropout: float = 0.1    # LoRA dropout probability

    # ------------------------------------------------------------------
    # Quantization (QLoRA / bitsandbytes)
    # ------------------------------------------------------------------
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"   # use "bfloat16" only on Ampere+ GPUs
    bnb_4bit_quant_type: str = "nf4"          # "fp4" or "nf4"
    use_nested_quant: bool = False            # double quantization

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    output_dir: str = "./results"
    num_train_epochs: int = 1
    fp16: bool = False
    bf16: bool = False  # keep False on GPUs with compute capability < 8 (e.g. T4)

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True

    max_grad_norm: float = 0.3
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "constant"
    max_steps: int = -1
    warmup_ratio: float = 0.03
    group_by_length: bool = True

    save_steps: int = 25
    logging_steps: int = 25
    report_to: str = "tensorboard"

    # ------------------------------------------------------------------
    # Sequence / packing
    # ------------------------------------------------------------------
    max_seq_length: int = 64   # keep explicit — None falls back to tokenizer defaults
    packing: bool = False

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device_map: Dict[str, int] = field(default_factory=lambda: {"": 0})


# Default instance matching the original notebook's settings.
default_config = FineTuneConfig()
