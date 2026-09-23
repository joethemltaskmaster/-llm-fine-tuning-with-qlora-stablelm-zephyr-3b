"""
utils.py — shared helpers used across the training and inference
pipelines.

Includes GPU memory management utilities (referenced by train.py) and
a logging setup helper for consistent output formatting across entry
points.
"""

import logging
import os

import torch


def set_cuda_alloc_conf(value: str = "expandable_segments:True") -> None:
    """
    Set PYTORCH_CUDA_ALLOC_CONF to reduce CUDA memory fragmentation.

    Helps avoid spurious OutOfMemoryError crashes on GPUs with limited
    VRAM (e.g. a 16 GB T4) where reserved-but-unallocated memory can't
    be reused for a new allocation due to fragmentation.

    Parameters
    ----------
    value : str
        The value to assign to PYTORCH_CUDA_ALLOC_CONF. Defaults to
        "expandable_segments:True".
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value


def clear_gpu_memory() -> None:
    """
    Run garbage collection and empty the CUDA memory cache.

    Call before starting a training or generation run (and between
    repeated runs in the same process) to release memory held by
    tensors that are no longer referenced.
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory_summary() -> str:
    """
    Return a short human-readable summary of current CUDA memory usage,
    useful for debugging OutOfMemoryError issues.

    Returns
    -------
    str
        Summary string, or a message indicating no CUDA device is
        available.
    """
    if not torch.cuda.is_available():
        return "No CUDA device available."

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    return (
        f"GPU memory — allocated: {allocated:.2f} GiB, "
        f"reserved: {reserved:.2f} GiB, total: {total:.2f} GiB"
    )


def setup_logging(level: int = logging.INFO, name: str = "llm_finetune") -> logging.Logger:
    """
    Configure and return a logger with consistent formatting across
    the training and inference entry points.

    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO).
    name : str
        Logger name. Defaults to "llm_finetune".

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False

    return logger
