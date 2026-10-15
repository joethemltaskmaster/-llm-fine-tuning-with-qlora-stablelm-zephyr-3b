"""
tests/test_model.py — smoke tests for model/tokenizer/LoRA setup.

These run on CPU with no downloads: `from_pretrained` is replaced with
fakes, so what's verified is *our* wiring (config values reaching the
right arguments, tokenizer padding setup, etc.) plus an environment
check that catches dependency regressions like the bitsandbytes
version error from the original notebook.

A real load of the quantized model (needs a CUDA GPU and network) is
opt-in:

    RUN_GPU_TESTS=1 pytest tests/test_model.py
"""

import dataclasses
import importlib.metadata as md
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import BitsAndBytesConfig

from llm_finetune import model as model_module
from llm_finetune.config import default_config
from llm_finetune.model import (
    build_bnb_config,
    build_lora_config,
    check_gpu_bf16_support,
    load_model,
    load_model_and_tokenizer,
    load_tokenizer,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeTokenizer:
    def __init__(self):
        self.eos_token = "</s>"
        self.pad_token = None
        self.padding_side = "left"
        self.added_special_tokens = None

    def add_special_tokens(self, tokens):
        self.added_special_tokens = tokens
        return 1


@pytest.fixture
def fake_model_loading(monkeypatch):
    """Replace HF from_pretrained calls; record the arguments they receive."""
    record = {}

    def fake_model_from_pretrained(name, **kwargs):
        record["model_name"] = name
        record["model_kwargs"] = kwargs
        record["model"] = SimpleNamespace(
            config=SimpleNamespace(use_cache=True, pretraining_tp=2)
        )
        return record["model"]

    def fake_tokenizer_from_pretrained(name, **kwargs):
        record["tokenizer_name"] = name
        record["tokenizer_kwargs"] = kwargs
        record["tokenizer"] = FakeTokenizer()
        return record["tokenizer"]

    monkeypatch.setattr(
        model_module.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained
    )
    monkeypatch.setattr(
        model_module.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained
    )
    return record


# ---------------------------------------------------------------------------
# Environment / dependency sanity (catches version regressions early)
# ---------------------------------------------------------------------------

def test_core_libraries_are_importable():
    import peft  # noqa: F401
    import transformers  # noqa: F401
    import trl  # noqa: F401


def test_bitsandbytes_meets_minimum_version():
    """transformers' 4-bit quantizer refuses bitsandbytes older than 0.46.1."""
    try:
        installed = md.version("bitsandbytes")
    except md.PackageNotFoundError:
        pytest.skip("bitsandbytes not installed in this environment")

    from packaging.version import Version

    assert Version(installed) >= Version("0.46.1"), (
        f"bitsandbytes {installed} is too old for 4-bit loading; "
        "run: pip install -U 'bitsandbytes>=0.46.1'"
    )


# ---------------------------------------------------------------------------
# build_bnb_config
# ---------------------------------------------------------------------------

def test_bnb_config_matches_default_config():
    bnb = build_bnb_config()
    assert isinstance(bnb, BitsAndBytesConfig)
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == default_config.bnb_4bit_quant_type
    assert bnb.bnb_4bit_use_double_quant == default_config.use_nested_quant
    assert bnb.bnb_4bit_compute_dtype == torch.float16


def test_bnb_config_honors_custom_values():
    cfg = dataclasses.replace(
        default_config,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype="bfloat16",
        use_nested_quant=True,
    )
    bnb = build_bnb_config(cfg)
    assert bnb.bnb_4bit_quant_type == "fp4"
    assert bnb.bnb_4bit_compute_dtype == torch.bfloat16
    assert bnb.bnb_4bit_use_double_quant is True


def test_bnb_config_invalid_dtype_name_raises():
    cfg = dataclasses.replace(default_config, bnb_4bit_compute_dtype="not_a_dtype")
    with pytest.raises(AttributeError):
        build_bnb_config(cfg)


# ---------------------------------------------------------------------------
# build_lora_config
# ---------------------------------------------------------------------------

def test_lora_config_matches_default_config():
    lora = build_lora_config()
    assert lora.r == default_config.lora_r
    assert lora.lora_alpha == default_config.lora_alpha
    assert lora.lora_dropout == default_config.lora_dropout
    assert lora.bias == "none"
    assert str(lora.task_type).endswith("CAUSAL_LM")


def test_lora_config_honors_custom_values():
    cfg = dataclasses.replace(default_config, lora_r=16, lora_alpha=32, lora_dropout=0.05)
    lora = build_lora_config(cfg)
    assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (16, 32, 0.05)


# ---------------------------------------------------------------------------
# load_tokenizer
# ---------------------------------------------------------------------------

def test_tokenizer_padding_setup(fake_model_loading):
    tok = load_tokenizer()
    assert tok.pad_token == tok.eos_token
    assert tok.padding_side == "right"
    assert tok.added_special_tokens == {"pad_token": "[PAD]"}


def test_tokenizer_loaded_from_configured_model_name(fake_model_loading):
    cfg = dataclasses.replace(default_config, model_name="some/other-model")
    load_tokenizer(cfg)
    assert fake_model_loading["tokenizer_name"] == "some/other-model"
    assert fake_model_loading["tokenizer_kwargs"].get("trust_remote_code") is True


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

def test_load_model_passes_quantization_and_device_map(fake_model_loading):
    load_model()
    kwargs = fake_model_loading["model_kwargs"]

    assert fake_model_loading["model_name"] == default_config.model_name
    assert isinstance(kwargs["quantization_config"], BitsAndBytesConfig)
    assert kwargs["device_map"] == default_config.device_map


def test_load_model_disables_cache_and_sets_tp(fake_model_loading):
    model = load_model()
    assert model.config.use_cache is False
    assert model.config.pretraining_tp == 1


# ---------------------------------------------------------------------------
# load_model_and_tokenizer
# ---------------------------------------------------------------------------

def test_load_model_and_tokenizer_returns_triple(fake_model_loading):
    model, tokenizer, peft_config = load_model_and_tokenizer()

    assert model is fake_model_loading["model"]
    assert tokenizer is fake_model_loading["tokenizer"]
    assert peft_config.r == default_config.lora_r


# ---------------------------------------------------------------------------
# check_gpu_bf16_support
# ---------------------------------------------------------------------------

def test_bf16_hint_printed_on_ampere_gpu(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))

    check_gpu_bf16_support()

    assert "supports bfloat16" in capsys.readouterr().out


def test_no_bf16_hint_on_t4(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))

    check_gpu_bf16_support()

    assert capsys.readouterr().out == ""


def test_bf16_check_is_safe_without_cuda(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    check_gpu_bf16_support()  # must not raise on CPU-only machines

    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Real quantized load (opt-in: needs CUDA + network)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="Set RUN_GPU_TESTS=1 on a CUDA machine to load the real model.",
)
def test_real_model_loads_in_4bit():
    model, tokenizer, _ = load_model_and_tokenizer()
    assert tokenizer.pad_token is not None
    assert getattr(model, "is_loaded_in_4bit", False)
