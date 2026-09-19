"""
model.py — quantized model loading, tokenizer setup, and LoRA configuration.

Wraps the model/tokenizer/LoRA construction logic so train.py and
inference.py can share a single source of truth for how the base
model is loaded and adapted.
"""

from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig

from llm_finetune.config import FineTuneConfig, default_config


def build_bnb_config(config: Optional[FineTuneConfig] = None) -> BitsAndBytesConfig:
    """
    Build the 4-bit quantization config used to load the base model.

    Parameters
    ----------
    config : FineTuneConfig, optional
        Defaults to `default_config` if not provided.

    Returns
    -------
    BitsAndBytesConfig
    """
    cfg = config or default_config
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.use_nested_quant,
    )


def check_gpu_bf16_support(config: Optional[FineTuneConfig] = None) -> None:
    """
    Print a hint if the current GPU supports bfloat16 acceleration but
    the config isn't using it. Mirrors the compatibility check from the
    original notebook. Does not raise or modify config — informational
    only.

    Parameters
    ----------
    config : FineTuneConfig, optional
        Defaults to `default_config` if not provided.
    """
    cfg = config or default_config
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    if compute_dtype == torch.float16 and cfg.use_4bit and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: consider setting bf16=True for faster training")
            print("=" * 80)


def load_model(config: Optional[FineTuneConfig] = None) -> PreTrainedModel:
    """
    Load the base causal LM with 4-bit quantization applied.

    Parameters
    ----------
    config : FineTuneConfig, optional
        Defaults to `default_config` if not provided.

    Returns
    -------
    PreTrainedModel
        The quantized base model, ready for LoRA adaptation.
    """
    cfg = config or default_config
    bnb_config = build_bnb_config(cfg)

    check_gpu_bf16_support(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map=cfg.device_map,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model


def load_tokenizer(config: Optional[FineTuneConfig] = None) -> PreTrainedTokenizer:
    """
    Load and configure the tokenizer for training/inference: adds a
    padding token if missing and sets right-padding, matching the
    original notebook's setup.

    Parameters
    ----------
    config : FineTuneConfig, optional
        Defaults to `default_config` if not provided.

    Returns
    -------
    PreTrainedTokenizer
    """
    cfg = config or default_config

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def build_lora_config(config: Optional[FineTuneConfig] = None) -> LoraConfig:
    """
    Build the LoRA adapter configuration.

    Parameters
    ----------
    config : FineTuneConfig, optional
        Defaults to `default_config` if not provided.

    Returns
    -------
    LoraConfig
    """
    cfg = config or default_config

    return LoraConfig(
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        r=cfg.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(
    config: Optional[FineTuneConfig] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, LoraConfig]:
    """
    Convenience function that loads the model, tokenizer, and LoRA
    config together in one call.

    Parameters
    ----------
    config : FineTuneConfig, optional
        Defaults to `default_config` if not provided.

    Returns
    -------
    tuple
        (model, tokenizer, peft_config)
    """
    cfg = config or default_config

    model = load_model(cfg)
    tokenizer = load_tokenizer(cfg)
    peft_config = build_lora_config(cfg)

    return model, tokenizer, peft_config
