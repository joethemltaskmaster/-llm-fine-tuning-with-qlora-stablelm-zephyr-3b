# llm-fine-tuning-with-qlora—stablelm-zephyr-3b

Fine-tunes [`stabilityai/stablelm-zephyr-3b`](https://huggingface.co/stabilityai/stablelm-zephyr-3b) on the [`mlabonne/guanaco-llama2-1k`](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) instruction dataset using 4-bit QLoRA, via Hugging Face `transformers`, `peft`, and `trl`.

Originally developed as a Google Colab notebook; this repo restructures it into an installable, testable Python package.

## Features

- 4-bit quantized base model loading (`bitsandbytes`, NF4)
- LoRA adapters for parameter-efficient fine-tuning (`peft`)
- Supervised fine-tuning loop via `trl.SFTTrainer`
- Gradient checkpointing and paged AdamW optimizer for reduced VRAM usage
- Streaming text generation for quick qualitative testing after training
- Config-driven — no need to edit source to change hyperparameters

## Hardware requirements

- Tested on a single **NVIDIA T4 (16 GB)** — Google Colab's free-tier GPU
- Works on any CUDA GPU with ~15 GB+ free VRAM given the default config (`max_seq_length=64`, `per_device_train_batch_size=1`, 4-bit quantization)
- `bf16` is disabled by default since T4 (compute capability 7.5) does not natively support bfloat16; enable it only on Ampere+ GPUs (A100, RTX 30/40 series)

## Project structure

```
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── src/
│   └── llm_finetune/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── inference.py
│       └── utils.py
├── scripts/
│   ├── run_train.sh
│   └── run_inference.py
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   └── test_model.py
└── notebooks/
    └── original_colab.ipynb
```

## Installation

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -U pip
pip install -e .
```

If you hit a `tokenizers` build failure on a fresh environment (common on Colab or bleeding-edge Python), it's almost always due to a missing prebuilt wheel forcing a source build, which needs Rust. See [Troubleshooting](#troubleshooting).

## Configuration

All hyperparameters live in `src/llm_finetune/config.py` (or an overriding YAML file, if you add one). Key settings:

| Parameter | Default | Notes |
|---|---|---|
| `model_name` | `stabilityai/stablelm-zephyr-3b` | Base model to fine-tune |
| `dataset_name` | `mlabonne/guanaco-llama2-1k` | Instruction dataset |
| `use_4bit` | `True` | 4-bit quantization via bitsandbytes |
| `bnb_4bit_compute_dtype` | `float16` | Use `bfloat16` only on Ampere+ GPUs |
| `lora_r` / `lora_alpha` / `lora_dropout` | `8` / `16` / `0.1` | LoRA adapter dimensions |
| `max_seq_length` | `64` | Increase only if VRAM allows — quadratic memory cost |
| `per_device_train_batch_size` | `1` | Increase with caution on T4 |
| `gradient_accumulation_steps` | `4` | Effective batch size = this × per-device batch size |
| `gradient_checkpointing` | `True` | Trades compute for memory |
| `optim` | `paged_adamw_32bit` | Reduces optimizer memory spikes |

## Usage

### Train

```bash
bash scripts/run_train.sh
```

This sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce CUDA memory fragmentation, then runs the training loop. Checkpoints and logs are written to `results/`.

### Monitor training

```bash
tensorboard --logdir results/runs
```

### Run inference

```bash
python scripts/run_inference.py --prompt "Tell me about operations research."
```

Streams the model's response token-by-token using the fine-tuned LoRA adapter.

## Testing

```bash
pytest tests/
```

Tests validate config consistency, dataset loading, and model/tokenizer initialization without requiring a full training run.

## Troubleshooting

**`ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes>=0.46.1`**
Older pinned versions of `bitsandbytes` don't support newer `transformers` quantizer APIs. Upgrade:
```bash
pip install -U "bitsandbytes>=0.46.1"
```

**`Failed building wheel for tokenizers`**
No prebuilt wheel exists for your Python version/platform, so pip falls back to a Rust source build. Fix by upgrading pip first (usually resolves it by finding a compatible wheel):
```bash
pip install --upgrade pip
pip install -U transformers tokenizers
```
If it still fails, install Rust:
```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
```

**`CUDA out of memory`**
On a T4, try in this order: reduce `per_device_train_batch_size`, enable `gradient_checkpointing`, lower `max_seq_length`, or increase `gradient_accumulation_steps` to compensate for a smaller batch size. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already set in `run_train.sh`) helps with fragmentation but won't fix a genuinely oversized model/batch.

## License

Add your license here.
