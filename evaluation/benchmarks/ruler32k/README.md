# RULER 32K

Synthetic long-context retrieval benchmark with 13 tasks and a target context
length of 32,768 tokens. Hugging Face dataset:
`xAlg-AI/att-hub-ruler-32k` (200 examples per task).

## Cache the dataset

```bash
export HF_HOME="$HOME/.cache/huggingface"
python evaluation/benchmarks/ruler32k/prepare_dataset.py
```

## Hugging Face layout

Each task has a same-named configuration and split:

```python
load_dataset("xAlg-AI/att-hub-ruler-32k", "cwe", split="cwe")
```

Columns: `context`, `question`, `answer_prefix`, `answer`, `task`, and
`max_new_tokens`. Tasks: `cwe`, `fwe`, `niah_multikey_1`, `niah_multikey_2`,
`niah_multikey_3`, `niah_multiquery`, `niah_multivalue`, `niah_single_1`,
`niah_single_2`, `niah_single_3`, `qa_1`, `qa_2`, and `vt`.
