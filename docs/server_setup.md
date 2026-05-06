# RunPod Server Setup

## Recommended Container

Use a PyTorch CUDA image when creating the pod. After SSH login, verify the GPU:

```bash
nvidia-smi
python --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Environment Variables

Put Hugging Face caches under `/workspace` so they survive restarts when the volume
is persistent:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_HUB_ENABLE_HF_TRANSFER=1
```

If you have a token:

```bash
export HF_TOKEN=<your_token>
```

## uv Setup

```bash
cd /workspace
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone <repo-url> project
cd project
uv sync --extra dev --extra quantization
uv run python -m pytest
```

## pip/venv Setup

```bash
cd /workspace/project
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,quantization]"
python -m pytest
```

When using Makefile targets from a venv, override the Python command:

```bash
make PYTHON=python qwen1_5b-dry-run
```

## Main 32GB Run

Start with dry-run:

```bash
make qwen1_5b-dry-run
```

Then run stages one by one so failures are easy to isolate:

```bash
make qwen1_5b-test
make qwen1_5b-run
make qwen1_5b-eval
```

Or run everything:

```bash
make qwen1_5b-all
```

## Downloading Results

Package results:

```bash
make package-results
```

Download the emitted `results_*.tar.gz` through VS Code Remote SSH. The archive
keeps configs, metrics, logs, predictions, and final model folders. Intermediate
checkpoint folders are excluded to keep the archive smaller.
