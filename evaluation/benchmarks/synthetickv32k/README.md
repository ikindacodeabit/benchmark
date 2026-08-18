# Synthetic-KV 32K

One compact key-value context containing 1,182 aligned retrieval questions.
The context has 31,945 Qwen3-8B tokens. Hugging Face dataset:
`ollamaweights/synthetic-dataset-1208`.

## What to use

The dataset is already generated and published on Hugging Face. You do not
need to generate or upload it.

Use `prepare_dataset.py` to download/cache the published dataset:

```bash
export HF_HOME="$HOME/.cache/huggingface"
python evaluation/benchmarks/synthetickv32k/prepare_dataset.py
```

Validate the existing cache without network access:

```bash
python evaluation/benchmarks/synthetickv32k/prepare_dataset.py --offline-check
```

Force a fresh Hugging Face download:

```bash
python evaluation/benchmarks/synthetickv32k/prepare_dataset.py --force-redownload
```

Do not use `generate_dataset.py` for normal evaluation setup. It is only for
maintainers who intentionally want to recreate and republish the dataset.

`download_dataset.sh` performs the same caching operation but contains
machine-specific paths. Use `prepare_dataset.py` unless you are on the machine
for which that shell script was written.

## Hugging Face layout

Configuration `default`, split `test`, with one row:

```python
load_dataset("ollamaweights/synthetic-dataset-1208", split="test")
```

Columns: `context_id`, `context`, `questions`, `answers`, `answer_prefix`,
`num_pairs`, `context_tokens`, and `max_new_tokens`. `questions[i]` maps to
`answers[i]`; KVPress expands the arrays into 1,182 evaluation rows.
