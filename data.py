"""
data.py — dataset loading and formatting.

Wraps `datasets.load_dataset` for the Guanaco instruction dataset and
keeps any prompt-formatting logic isolated from the training loop in
train.py.

The `mlabonne/guanaco-llama2-1k` dataset ships pre-formatted examples
under a single "text" column (Llama-2 style `<s>[INST] ... [/INST] ...`
strings), so no extra templating is required by default. The
formatting helpers below are provided for cases where you swap in a
raw/unformatted dataset instead.
"""

from typing import Optional

from datasets import load_dataset, Dataset

from llm_finetune.config import FineTuneConfig, default_config


def load_training_dataset(config: Optional[FineTuneConfig] = None) -> Dataset:
    """
    Load the training split of the configured instruction dataset.

    Parameters
    ----------
    config : FineTuneConfig, optional
        Configuration object specifying `dataset_name`. Defaults to
        `default_config` if not provided.

    Returns
    -------
    Dataset
        The "train" split, ready to pass to SFTTrainer.
    """
    cfg = config or default_config
    dataset = load_dataset(cfg.dataset_name, split="train")
    return dataset


def format_instruction_prompt(instruction: str, response: str = "") -> str:
    """
    Format a raw instruction/response pair into the Llama-2-style
    prompt template used by this project's inference/training flow.

    Use this only if working with a raw dataset that isn't already
    pre-formatted (unlike `mlabonne/guanaco-llama2-1k`, which ships
    ready-to-use "text" fields).

    Parameters
    ----------
    instruction : str
        The user instruction / prompt text.
    response : str, optional
        The target completion. Leave empty when formatting a prompt
        for inference rather than training.

    Returns
    -------
    str
        The formatted prompt string.
    """
    system_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    )
    b_inst, e_inst = "### Instruction:\n", "### Response:\n"

    prompt = f"{system_prompt}{b_inst}{instruction.strip()}\n\n{e_inst}"
    if response:
        prompt += response.strip()
    return prompt


def format_dataset(dataset: Dataset, instruction_col: str = "instruction",
                    response_col: str = "response") -> Dataset:
    """
    Apply `format_instruction_prompt` across a raw dataset to produce a
    single "text" column, matching what SFTTrainer expects.

    Only needed for datasets that aren't already pre-formatted. Not
    required for the default `mlabonne/guanaco-llama2-1k` dataset.

    Parameters
    ----------
    dataset : Dataset
        A dataset with separate instruction/response columns.
    instruction_col : str
        Name of the column containing the instruction text.
    response_col : str
        Name of the column containing the target response text.

    Returns
    -------
    Dataset
        Dataset with a single formatted "text" column.
    """
    def _format(example):
        return {
            "text": format_instruction_prompt(
                example[instruction_col], example[response_col]
            )
        }

    return dataset.map(_format)


def inspect_sequence_lengths(dataset: Dataset, tokenizer, text_col: str = "text",
                              sample_size: Optional[int] = None) -> dict:
    """
    Compute token-length statistics for a dataset's text column, useful
    for picking a `max_seq_length` that covers most examples without
    wasting VRAM on unnecessarily long sequences.

    Parameters
    ----------
    dataset : Dataset
        Dataset containing a text column to measure.
    tokenizer
        A Hugging Face tokenizer instance.
    text_col : str
        Name of the column containing text to tokenize.
    sample_size : int, optional
        If set, only inspect the first N examples (useful for large
        datasets where tokenizing everything is slow).

    Returns
    -------
    dict
        Dictionary with "min", "max", "mean" token lengths.
    """
    examples = dataset if sample_size is None else dataset.select(range(min(sample_size, len(dataset))))
    lengths = [len(tokenizer(ex[text_col])["input_ids"]) for ex in examples]

    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }
