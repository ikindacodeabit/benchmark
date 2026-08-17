# Synthetic-KV 32K

- Hugging Face dataset: `ollamaweights/synthetic-dataset-1208`
- Evaluation name: `synthetic_kv_32k`

The dataset is published on Hugging Face; `prepare_dataset.py` is the local
creation/download entry point and validates the compact `test` split.

From the repository root, prepare and offline-validate this dataset:

```bash
bash evaluation/benchmarks/synthetic32k/download_dataset.sh
```
