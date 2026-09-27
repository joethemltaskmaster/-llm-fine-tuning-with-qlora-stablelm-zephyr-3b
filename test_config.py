"""
tests/test_config.py — sanity checks for FineTuneConfig.

Validates that FineTuneConfig.validate() correctly catches internally
inconsistent or invalid settings (e.g. bf16 and fp16 both enabled, a
non-positive max_seq_length) *before* any GPU time is spent, rather
than failing mid-training.
"""

import dataclasses

import pytest

from llm_finetune.config import FineTuneConfig, default_config


# ---------------------------------------------------------------------------
# The default config itself must always be valid
# ---------------------------------------------------------------------------

def test_default_config_is_valid():
    default_config.validate()  # should not raise


def test_default_config_does_not_enable_both_precisions():
    assert not (default_config.fp16 and default_config.bf16)


def test_default_max_seq_length_is_positive():
    assert default_config.max_seq_length > 0


# ---------------------------------------------------------------------------
# Precision flags
# ---------------------------------------------------------------------------

def test_bf16_and_fp16_both_true_is_invalid():
    cfg = dataclasses.replace(default_config, fp16=True, bf16=True)
    with pytest.raises(ValueError, match="fp16 and bf16"):
        cfg.validate()


@pytest.mark.parametrize("fp16,bf16", [(True, False), (False, True), (False, False)])
def test_at_most_one_precision_flag_is_valid(fp16, bf16):
    cfg = dataclasses.replace(default_config, fp16=fp16, bf16=bf16)
    cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# Sequence length
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [0, -1, -64])
def test_non_positive_max_seq_length_is_invalid(bad_value):
    cfg = dataclasses.replace(default_config, max_seq_length=bad_value)
    with pytest.raises(ValueError, match="max_seq_length"):
        cfg.validate()


def test_positive_max_seq_length_is_valid():
    cfg = dataclasses.replace(default_config, max_seq_length=512)
    cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# LoRA hyperparameters
# ---------------------------------------------------------------------------

def test_lora_r_is_positive_by_default():
    assert default_config.lora_r > 0


@pytest.mark.parametrize("bad_r", [0, -1, -8])
def test_non_positive_lora_r_is_invalid(bad_r):
    cfg = dataclasses.replace(default_config, lora_r=bad_r)
    with pytest.raises(ValueError, match="lora_r"):
        cfg.validate()


@pytest.mark.parametrize("bad_dropout", [-0.1, 1.0, 1.5])
def test_lora_dropout_outside_valid_range_is_invalid(bad_dropout):
    cfg = dataclasses.replace(default_config, lora_dropout=bad_dropout)
    with pytest.raises(ValueError, match="lora_dropout"):
        cfg.validate()


@pytest.mark.parametrize("ok_dropout", [0.0, 0.1, 0.5, 0.99])
def test_lora_dropout_inside_valid_range_is_valid(ok_dropout):
    cfg = dataclasses.replace(default_config, lora_dropout=ok_dropout)
    cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# Batch size / gradient accumulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field_name", [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
])
def test_non_positive_batch_related_fields_are_invalid(field_name):
    cfg = dataclasses.replace(default_config, **{field_name: 0})
    with pytest.raises(ValueError, match=field_name):
        cfg.validate()


# ---------------------------------------------------------------------------
# Learning rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_lr", [0.0, -2e-4])
def test_non_positive_learning_rate_is_invalid(bad_lr):
    cfg = dataclasses.replace(default_config, learning_rate=bad_lr)
    with pytest.raises(ValueError, match="learning_rate"):
        cfg.validate()


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def test_default_quant_type_is_recognized():
    assert default_config.bnb_4bit_quant_type in {"fp4", "nf4"}


def test_unrecognized_quant_type_is_invalid():
    cfg = dataclasses.replace(default_config, bnb_4bit_quant_type="invalid")
    with pytest.raises(ValueError, match="bnb_4bit_quant_type"):
        cfg.validate()


@pytest.mark.parametrize("ok_type", ["fp4", "nf4"])
def test_recognized_quant_types_are_valid(ok_type):
    cfg = dataclasses.replace(default_config, bnb_4bit_quant_type=ok_type)
    cfg.validate()  # should not raise
