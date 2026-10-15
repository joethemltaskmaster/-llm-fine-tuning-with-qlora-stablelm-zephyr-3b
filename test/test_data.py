"""
tests/test_data.py — checks for dataset loading and prompt formatting.

Most tests run offline against small in-memory datasets and fakes, so
they are fast and need no network or GPU. One optional test hits the
real Hugging Face dataset to confirm its schema before you burn GPU
time on a broken training run; enable it with:

    RUN_NETWORK_TESTS=1 pytest tests/test_data.py
"""

import dataclasses
import os

import pytest
from datasets import Dataset

from llm_finetune import data as data_module
from llm_finetune.config import default_config
from llm_finetune.data import (
    format_dataset,
    format_instruction_prompt,
    inspect_sequence_lengths,
    load_training_dataset,
)


class FakeTokenizer:
    """Whitespace tokenizer: one token per word. Mimics HF's dict output."""

    def __call__(self, text):
        return {"input_ids": text.split()}


# ---------------------------------------------------------------------------
# format_instruction_prompt
# ---------------------------------------------------------------------------

def test_prompt_contains_template_markers():
    prompt = format_instruction_prompt("Explain gradient descent.")
    assert "### Instruction:" in prompt
    assert "### Response:" in prompt


def test_prompt_starts_with_system_preamble():
    prompt = format_instruction_prompt("Hello")
    assert prompt.startswith("Below is an instruction that describes a task.")


def test_prompt_without_response_ends_at_response_marker():
    prompt = format_instruction_prompt("Hello")
    assert prompt.endswith("### Response:\n")


def test_prompt_strips_instruction_whitespace():
    prompt = format_instruction_prompt("   padded instruction \n")
    assert "### Instruction:\npadded instruction\n\n" in prompt


def test_prompt_appends_stripped_response_after_marker():
    prompt = format_instruction_prompt("Q", "  the answer  \n")
    assert prompt.endswith("### Response:\nthe answer")


def test_instruction_comes_before_response():
    prompt = format_instruction_prompt("my question", "my answer")
    assert prompt.index("my question") < prompt.index("my answer")


# ---------------------------------------------------------------------------
# format_dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_dataset():
    return Dataset.from_dict(
        {
            "instruction": ["What is 2+2?", "Name a color."],
            "response": ["4", "Blue"],
        }
    )


def test_format_dataset_adds_text_column(raw_dataset):
    formatted = format_dataset(raw_dataset)
    assert "text" in formatted.column_names
    assert len(formatted) == len(raw_dataset)


def test_format_dataset_text_contains_instruction_and_response(raw_dataset):
    formatted = format_dataset(raw_dataset)
    for row, original in zip(formatted, raw_dataset):
        assert original["instruction"] in row["text"]
        assert original["response"] in row["text"]


def test_format_dataset_supports_custom_column_names():
    ds = Dataset.from_dict({"prompt": ["Hi"], "completion": ["Hello there"]})
    formatted = format_dataset(ds, instruction_col="prompt", response_col="completion")
    assert "Hi" in formatted[0]["text"]
    assert "Hello there" in formatted[0]["text"]


def test_format_dataset_missing_column_raises(raw_dataset):
    with pytest.raises(KeyError):
        format_dataset(raw_dataset, instruction_col="does_not_exist")


# ---------------------------------------------------------------------------
# inspect_sequence_lengths
# ---------------------------------------------------------------------------

@pytest.fixture
def text_dataset():
    # Token counts with FakeTokenizer: 3, 5, 1
    return Dataset.from_dict({"text": ["a b c", "a b c d e", "a"]})


def test_inspect_sequence_lengths_stats(text_dataset):
    stats = inspect_sequence_lengths(text_dataset, FakeTokenizer())
    assert stats["min"] == 1
    assert stats["max"] == 5
    assert stats["mean"] == pytest.approx(3.0)


def test_inspect_sequence_lengths_respects_sample_size(text_dataset):
    # First two examples only: lengths 3 and 5
    stats = inspect_sequence_lengths(text_dataset, FakeTokenizer(), sample_size=2)
    assert stats["min"] == 3
    assert stats["max"] == 5
    assert stats["mean"] == pytest.approx(4.0)


def test_inspect_sequence_lengths_sample_larger_than_dataset(text_dataset):
    stats = inspect_sequence_lengths(text_dataset, FakeTokenizer(), sample_size=1000)
    assert stats["max"] == 5


def test_inspect_sequence_lengths_custom_text_column():
    ds = Dataset.from_dict({"body": ["one two", "one"]})
    stats = inspect_sequence_lengths(ds, FakeTokenizer(), text_col="body")
    assert stats["max"] == 2


# ---------------------------------------------------------------------------
# load_training_dataset (offline: load_dataset is faked)
# ---------------------------------------------------------------------------

def test_load_training_dataset_uses_default_config(monkeypatch):
    calls = []

    def fake_load_dataset(name, split=None, **kwargs):
        calls.append((name, split))
        return "sentinel-dataset"

    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    result = load_training_dataset()

    assert result == "sentinel-dataset"
    assert calls == [(default_config.dataset_name, "train")]


def test_load_training_dataset_honors_custom_config(monkeypatch):
    calls = []
    monkeypatch.setattr(
        data_module,
        "load_dataset",
        lambda name, split=None, **kw: calls.append((name, split)) or "ds",
    )

    cfg = dataclasses.replace(default_config, dataset_name="some/other-dataset")
    load_training_dataset(cfg)

    assert calls == [("some/other-dataset", "train")]


# ---------------------------------------------------------------------------
# Real dataset schema check (opt-in: needs network)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="Set RUN_NETWORK_TESTS=1 to download the real dataset.",
)
def test_real_dataset_has_expected_schema():
    ds = load_training_dataset()
    assert len(ds) > 0
    assert "text" in ds.column_names, (
        f"Expected a 'text' column for SFTTrainer, found: {ds.column_names}"
    )
    sample = ds["text"][:50]
    assert all(isinstance(t, str) and t.strip() for t in sample)
