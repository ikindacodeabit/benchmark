# RULER 32K

This integration follows the
[sparse-attention-hub RULER32K benchmark](https://github.com/skylight-org/sparse-attention-hub/tree/main/benchmark/ruler32k).

- Dataset: `xAlg-AI/att-hub-ruler-32k`
- Context length: `32768`
- Samples: 200 per task
- Tasks: `cwe`, `fwe`, `niah_multikey_1`, `niah_multikey_2`,
  `niah_multikey_3`, `niah_multiquery`, `niah_multivalue`, `niah_single_1`,
  `niah_single_2`, `niah_single_3`, `qa_1`, `qa_2`, and `vt`

The dataset stores every task under a same-named Hugging Face configuration
and split. For example:

```python
load_dataset("xAlg-AI/att-hub-ruler-32k", "cwe", split="cwe")
```

The shared KVPress and direct-baseline loaders both use this contract. The
metric implementation in `calculate_metrics.py` matches sparse-attention-hub:
QA tasks use partial string match, while all other tasks use all-answer string
match.
