# RULER 64K

Synthetic long-context retrieval benchmark with 13 tasks and a target context
length of 65,536 tokens. Hugging Face dataset: `ollamaweights/Ruler-64k`
(6,500 examples; 500 per task).

## Cache the dataset

```bash
export HF_HOME="$HOME/.cache/huggingface"
python evaluation/benchmarks/ruler64k/prepare_dataset.py
```

## Hugging Face layout

All tasks are rows in configuration `65536`, split `test`:

```python
load_dataset("ollamaweights/Ruler-64k", "65536", split="test")
```

Columns: `context`, `question`, `answer_prefix`, `answer`, `task`, and
`max_new_tokens`. The `task` column selects `cwe`, `fwe`, the NIAH variants,
`qa_1`, `qa_2`, or `vt`.
