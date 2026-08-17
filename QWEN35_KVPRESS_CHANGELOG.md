# KVPress Qwen3.5 Adapter Changes

This document describes the uncommitted Qwen3.5 support changes. Existing
KVPress compression algorithms were not rewritten; model and cache handling
was added around them.

## Before and after

| Existing KVPress | With Qwen3.5 support |
|---|---|
| Assumes every decoder layer has `layer.self_attn` | Uses a model adapter to discover compatible layers |
| Hooks every decoder layer | Hooks only `full_attention` layers |
| Treats every layer as a normal KV cache layer | Preserves Qwen3.5 KV, recurrent, and convolution state separately |
| Counts KV memory for every hidden layer | Counts only full-attention layers for Qwen3.5 |
| Manually truncates generated tokens | Snapshots the context cache and restores it after every answer |
| Loads every checkpoint through the causal-LM pipeline | Adds a guarded text-only Qwen3.5 loader |

Qwen2, Qwen3, Llama, Mistral, Phi-3, and Gemma3 continue using the standard
behavior. KVzip has a separate custom integration described below.

## 1. New model adapter

File: [`kvpress/model_adapter.py`](kvpress/model_adapter.py)

The adapter interface is:

```python
class ModelAdapter:
    def get_text_config(self, model): ...
    def get_language_model(self, model): ...
    def iter_kv_attention_layers(self, model): ...
    def create_cache(self, model): ...
    def kv_bytes_per_token(self, model, batch_size=1): ...
    def snapshot_cache_state(self, cache): ...
    def restore_cache_state(self, cache, snapshot): ...
```

### StandardModelAdapter

This preserves the previous behavior:

```python
config = model.config
language_model = (
    model.model.language_model
    if hasattr(model.model, "language_model")
    else model.model
)
```

It iterates over every `layer.self_attn`, creates `DynamicCache()`, calculates
KV memory using all hidden layers, and restores normal `keys` and `values`.

### Qwen35ModelAdapter

Qwen3.5 is selected only when:

```python
model.config.model_type == "qwen3_5"
```

It uses:

```python
text_config = model.config.text_config
language_model = model.model.language_model
```

Only full-attention layers are yielded:

```python
for layer_idx, layer in enumerate(language_model.layers):
    if text_config.layer_types[layer_idx] == "full_attention":
        yield layer_idx, layer.self_attn
```

Linear-attention/DeltaNet layers never receive KVPress hooks. KV memory uses:

```python
num_kv_layers = text_config.layer_types.count("full_attention")
```

The installed Transformers version provides `Qwen3_5DynamicCache` with these
exact fields, which are preserved during multi-question inference:

```text
key_cache
value_cache
recurrent_states
conv_states
layer_types
transformer_layers
last_linear_layer
cache metadata
```

## 2. Press hook changes

File: [`kvpress/presses/base_press.py`](kvpress/presses/base_press.py)

The old direct layer loop was replaced with adapter discovery:

```python
adapter = get_model_adapter(model)

for layer_idx, attention in adapter.iter_kv_attention_layers(model):
    attention.rotary_emb = adapter.get_language_model(model).rotary_emb
    attention.layer_idx = layer_idx
    hooks.append(
        attention.register_forward_hook(
            self.forward_hook,
            with_kwargs=True,
        )
    )
```

Cache access is also adapter-specific:

```python
keys, values = adapter.get_keys_and_values(cache, module.layer_idx)
keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)
adapter.set_keys_and_values(cache, module.layer_idx, keys, values)
```

For ordinary `BasePress` subclasses, Qwen3.5 initially allows `no_press`,
`random`, and `knorm`. Other ordinary presses raise a clear
`NotImplementedError` until their query-projection handling is implemented.
KVzip is an exception because it owns a custom reconstruction and hook path;
that path now uses the adapter's layer and cache accessors and maps score rows
only to full-attention layers.

## 3. Pipeline changes

File: [`kvpress/pipeline.py`](kvpress/pipeline.py)

KV memory now comes from the adapter:

```python
return get_model_adapter(self.model).kv_bytes_per_token(self.model, batch_size)
```

Cache creation now uses the model adapter:

```python
adapter = get_model_adapter(self.model)
if cache is None:
    cache = adapter.create_cache(self.model)
```

For questions sharing a context:

```python
context_snapshot = adapter.snapshot_cache_state(cache)
```

After every generated answer:

```python
adapter.restore_cache_state(cache, context_snapshot)
```

The existing `_remove_answer_from_cache` helper remains for compatibility but
delegates to the selected adapter.

## 4. Evaluation and loading changes

File: [`evaluation/evaluate.py`](evaluation/evaluate.py)

Qwen3.5 checkpoints are detected through `AutoConfig`. They are loaded with
`Qwen3_5ForConditionalGeneration` and `AutoTokenizer`, then passed directly to
`KVPressTextGenerationPipeline`. No image or video inputs are supplied.

Existing models retain the original `transformers.pipeline(...)` loading path.

Attention state reset now uses:

```python
adapter = get_model_adapter(self.pipeline.model)
for _, attention in adapter.iter_kv_attention_layers(self.pipeline.model):
    attention.masked_key_indices = None
```

## 5. Query/key normalization

File: [`kvpress/utils.py`](kvpress/utils.py)

`Qwen3_5Attention` is included with Qwen3 and Gemma3 when applying `q_norm`
and `k_norm` to pre-RoPE query/key states.

## 6. Tests

File: [`tests/test_model_adapter.py`](tests/test_model_adapter.py)

Tests cover:

- Qwen3 selecting the unchanged standard adapter.
- Qwen3.5 selecting the Qwen3.5 adapter.
- Discovery of exactly the `full_attention` layers.
- Exclusion of linear-attention layers from hooks.
- KV memory counting only full-attention layers.
- Restoration of attention, recurrent, and convolution cache state.

KVzip's custom path now also uses adapter layer iteration and cache access,
instead of assuming `cache.layers` or `layer.self_attn` on every decoder layer.

## 7. Synthetic-KV Qwen2.5 benchmark files

The downloaded model is stored locally at:

```text
Qwen2.5-7B-Instruct-1M/
```

Added benchmark files:

```text
benchmark_artifacts/
├── yml/synthetic_kv/
│   ├── 32k/all_tasks/qwen25_1m/evaluate_synthetic_kv_32k_qwen25_1m_baseline.yaml
│   └── 64k/all_tasks/qwen25_1m/evaluate_synthetic_kv_64k_qwen25_1m_baseline.yaml
└── slurm_jobs/synthetic_kv/
    ├── 32k/all_tasks/qwen25_1m/synthetic-kv-32k-qwen25-1m-baseline-l40.sh
    └── 64k/all_tasks/qwen25_1m/synthetic-kv-64k-qwen25-1m-baseline-l40.sh
```

Both use `no_press`, `max_context_length: null`, and one L40 GPU. Results are
written to:

```text
benchmark_artifacts/results/synthetic_kv/32k/runs/results_synthetic_kv_32k_qwen25_1m_baseline/
benchmark_artifacts/results/synthetic_kv/64k/runs/results_synthetic_kv_64k_qwen25_1m_baseline/
```

The large local model directory is excluded through the `Qwen2.5-*` entry in
[`.gitignore`](.gitignore).

## 8. Verification

Passed:

- Python byte-compilation of changed Python files.
- Shell syntax checks for both Slurm scripts.
- `git diff --check`.
- Focused adapter checks.

The full pytest suite was not run because `pytest` is not installed in the
`kvpress` environment. No git commit was created.
