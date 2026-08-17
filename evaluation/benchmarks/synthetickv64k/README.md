# Synthetic-KV 64K

- Hugging Face dataset: `ollamaweights/synthetickv_formated`
- Evaluation name: `synthetic_kv`

The local generation/publishing entry point is
`create_huggingface_dataset.py`. `prepare_dataset.py` downloads or validates
the published Hugging Face copy.

From the repository root, prepare and offline-validate this dataset:

```bash
bash evaluation/benchmarks/synthetickv64k/download_dataset.sh
```
