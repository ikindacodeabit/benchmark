#!/usr/bin/env python3
"""Verify that KVPress's Qwen3.5 adapter matches the installed Transformers API."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}", flush=True)
    if not condition:
        raise AssertionError(message)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import torch
    import transformers
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

    from kvpress.model_adapter import Qwen35ModelAdapter, StandardModelAdapter, get_model_adapter

    print(f"Repository: {repo_root}")
    print(f"Transformers: {transformers.__version__}")

    check(hasattr(transformers, "DynamicCache"), "Transformers exposes DynamicCache")
    check(Qwen3_5DynamicCache.__name__ == "Qwen3_5DynamicCache", "Qwen3.5 specialized cache exists")

    text_config = Qwen3_5TextConfig()
    cache = Qwen3_5DynamicCache(config=text_config)
    for field in ("key_cache", "value_cache", "recurrent_states", "conv_states", "transformer_layers"):
        check(hasattr(cache, field), f"Qwen3.5 cache has {field}")

    layers = [
        SimpleNamespace(self_attn="full-0"),
        SimpleNamespace(linear_attn="linear-1"),
        SimpleNamespace(self_attn="full-2"),
    ]
    qwen_model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="qwen3_5",
            text_config=SimpleNamespace(
                layer_types=["full_attention", "linear_attention", "full_attention"],
                num_key_value_heads=2,
                head_dim=64,
                num_hidden_layers=3,
            ),
        ),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
        dtype=torch.bfloat16,
    )
    adapter = get_model_adapter(qwen_model)
    check(isinstance(adapter, Qwen35ModelAdapter), "qwen3_5 selects Qwen35ModelAdapter")
    discovered = list(adapter.iter_kv_attention_layers(qwen_model))
    check(discovered == [(0, "full-0"), (2, "full-2")], "only full_attention layers are exposed")
    check(adapter.kv_bytes_per_token(qwen_model) == 2 * 2 * 2 * 64 * 2, "KV memory counts full-attention layers only")

    adapter_cache = adapter.create_cache(qwen_model)
    check(isinstance(adapter_cache, Qwen3_5DynamicCache), "adapter creates the installed Qwen3.5 cache")
    check(adapter_cache.transformer_layers == [0, 2], "cache transformer_layers matches full-attention layers")

    # Exercise the exact transformer_layers assumption used by
    # Qwen35ModelAdapter.cache_seq_lengths/truncate_cache.
    adapter_cache.key_cache[0] = torch.zeros(1, 2, 5, 64)
    adapter_cache.value_cache[0] = torch.zeros(1, 2, 5, 64)
    adapter_cache.key_cache[2] = torch.zeros(1, 2, 7, 64)
    adapter_cache.value_cache[2] = torch.zeros(1, 2, 7, 64)
    check(adapter.cache_seq_lengths(adapter_cache) == [5, 7], "full-attention cache lengths are readable")
    adapter.truncate_cache(adapter_cache, [3, 4])
    check(adapter_cache.key_cache[0].shape[-2] == 3, "first full-attention cache truncates correctly")
    check(adapter_cache.key_cache[2].shape[-2] == 4, "second full-attention cache truncates correctly")

    standard_model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_size=512,
            head_dim=64,
        ),
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn="standard-0"), SimpleNamespace(self_attn="standard-1")]
        ),
        dtype=torch.float16,
    )
    standard_adapter = get_model_adapter(standard_model)
    check(isinstance(standard_adapter, StandardModelAdapter), "qwen3 selects StandardModelAdapter")
    check(
        list(standard_adapter.iter_kv_attention_layers(standard_model))
        == [(0, "standard-0"), (1, "standard-1")],
        "standard adapter exposes every attention layer",
    )
    check(standard_adapter.kv_bytes_per_token(standard_model) == 2 * 2 * 2 * 64 * 2, "standard KV accounting is unchanged")

    print("\nAll Qwen3.5 adapter compatibility checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        print(f"\nVerification failed: {error}", flush=True)
        exit_code = 1
    # Transformers can leave optional kernel-loader threads alive in this
    # environment; use an explicit process exit after all checks are printed.
    os._exit(exit_code)
