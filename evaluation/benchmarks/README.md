# Benchmark dependencies

Python 3.10 or newer is required. Run commands from the KVPress repository
root so imports and relative paths resolve correctly.

## Recommended installation

Install the project and evaluation dependencies declared in `pyproject.toml`:

```bash
uv sync --extra eval
```

If `uv` is unavailable:

```bash
python -m pip install -e ".[eval]"
```

## Dataset-only tasks

Downloading, caching, and offline validation require `datasets`:

```bash
python -m pip install "datasets>=2.21.0"
```

Synthetic-KV generation additionally requires `transformers` because context
sizes are measured with the target tokenizer:

```bash
python -m pip install "transformers>=4.56.0,<5.3"
```

Uploading a regenerated dataset requires Hugging Face authentication:

```bash
huggingface-cli login
```

Normal benchmark setup does not require generation or upload. Use the
`prepare_dataset.py` command documented in each dataset folder.

## Environment variables

Set one shared cache location for login and compute nodes:

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

For offline compute nodes:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Model-specific environments, including Qwen3.5 GPTQ dependencies, are
documented in the `qwen35-gptq` section of `pyproject.toml`. Dataset caching
does not require loading a model or running evaluation.
