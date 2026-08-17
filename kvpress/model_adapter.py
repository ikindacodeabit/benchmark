"""Small model-specific shims used by KVPress.

The standard adapter intentionally mirrors the cache and layer handling that
KVPress used before hybrid Qwen3.5 models were added.  Qwen3.5 has a mixture
of full-attention and recurrent layers, so only its full-attention layers are
presented to KVPress.
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from transformers import DynamicCache, QuantizedCache
from transformers.models.llama.modeling_llama import rotate_half


class ModelAdapter:
    def get_text_config(self, model):
        raise NotImplementedError

    def get_language_model(self, model):
        raise NotImplementedError

    def iter_kv_attention_layers(self, model):
        raise NotImplementedError

    def create_cache(self, model):
        raise NotImplementedError

    def kv_bytes_per_token(self, model, batch_size=1):
        raise NotImplementedError

    def snapshot_cache_state(self, cache):
        raise NotImplementedError

    def restore_cache_state(self, cache, snapshot):
        raise NotImplementedError

    def cached_continuation(self, model):
        """Adapt a cached multi-token model call without changing its outer KVPress flow."""
        return contextlib.nullcontext()

    # KVzip compatibility hooks. The defaults preserve upstream KVzip behavior.
    def kvzip_apply_query_rope(self, queries: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))

    def kvzip_forward_kwargs(self) -> dict[str, int]:
        return {"num_logits_to_keep": 1}

    def kvzip_snapshot_reconstruction_state(self, cache):
        return None

    def kvzip_restore_reconstruction_state(self, cache, snapshot) -> None:
        return None

    # These helpers keep cache-specific details out of the press and pipeline.
    def get_keys_and_values(self, cache, layer_idx: int):
        layer = cache.layers[layer_idx]
        if isinstance(cache, QuantizedCache):
            return layer._dequantize(layer._quantized_keys), layer._dequantize(layer._quantized_values)
        return layer.keys, layer.values

    def set_keys_and_values(self, cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        layer = cache.layers[layer_idx]
        if isinstance(cache, QuantizedCache):
            layer._quantized_keys = layer._quantize(keys, axis=layer.axis_key)
            layer._quantized_values = layer._quantize(values, axis=layer.axis_value)
            layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            layer.values = torch.zeros(0, dtype=values.dtype, device=values.device)
            layer.cumulative_length = keys.shape[-2]
        else:
            layer.keys = keys
            layer.values = values

    def cache_seq_lengths(self, cache) -> list[int]:
        return [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]

    def truncate_cache(self, cache, cache_seq_lengths: list[int]) -> None:
        for layer_idx, sequence_length in enumerate(cache_seq_lengths):
            layer = cache.layers[layer_idx]
            if layer.keys is not None:
                layer.keys = layer.keys[:, :, :sequence_length]
                layer.values = layer.values[:, :, :sequence_length]
            if isinstance(cache, QuantizedCache):
                layer._quantized_keys = layer._quantized_keys[:, :, :sequence_length]
                layer._quantized_values = layer._quantized_values[:, :, :sequence_length]

    @property
    def is_qwen35(self) -> bool:
        return False


class StandardModelAdapter(ModelAdapter):
    def get_text_config(self, model):
        return model.config

    def get_language_model(self, model):
        return model.model.language_model if hasattr(model.model, "language_model") else model.model

    def iter_kv_attention_layers(self, model) -> Iterator[tuple[int, Any]]:
        for layer_idx, layer in enumerate(self.get_language_model(model).layers):
            yield layer_idx, layer.self_attn

    def create_cache(self, model):
        return DynamicCache()

    def kv_bytes_per_token(self, model, batch_size=1):
        config = self.get_text_config(model)
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        bytes_per_element = torch.finfo(model.dtype).bits // 8
        return num_layers * 2 * num_kv_heads * head_dim * bytes_per_element * batch_size

    def snapshot_cache_state(self, cache):
        layers = []
        for layer in cache.layers:
            state = {name: getattr(layer, name) for name in ("keys", "values") if hasattr(layer, name)}
            for name in ("_quantized_keys", "_quantized_values", "cumulative_length"):
                if hasattr(layer, name):
                    state[name] = getattr(layer, name)
            layers.append(state)
        metadata = {
            name: getattr(cache, name)
            for name in ("_seen_tokens", "seen_tokens", "cache_position")
            if hasattr(cache, name)
        }
        return {"layers": layers, "metadata": metadata}

    def restore_cache_state(self, cache, snapshot):
        for layer, state in zip(cache.layers, snapshot["layers"]):
            for name, value in state.items():
                setattr(layer, name, value)
        for name, value in snapshot["metadata"].items():
            setattr(cache, name, value)


class Qwen35ModelAdapter(ModelAdapter):
    @property
    def is_qwen35(self) -> bool:
        return True

    def get_text_config(self, model):
        config = model.config
        return config.text_config if hasattr(config, "text_config") else config

    def get_language_model(self, model):
        return model.model.language_model if hasattr(model.model, "language_model") else model.model

    def iter_kv_attention_layers(self, model) -> Iterator[tuple[int, Any]]:
        language_model = self.get_language_model(model)
        text_config = self.get_text_config(model)
        for layer_idx, layer in enumerate(language_model.layers):
            if text_config.layer_types[layer_idx] == "full_attention":
                yield layer_idx, layer.self_attn

    def create_cache(self, model):
        # Recent Transformers releases provide a hybrid cache with the exact
        # recurrent/convolution state fields used by Qwen3.5.  Fall back to
        # the requested DynamicCache construction on releases without it.
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

            return Qwen3_5DynamicCache(config=self.get_text_config(model))
        except (ImportError, TypeError):  # pragma: no cover - older Transformers
            return DynamicCache(config=self.get_text_config(model))

    def kv_bytes_per_token(self, model, batch_size=1):
        text_config = self.get_text_config(model)
        num_kv_layers = text_config.layer_types.count("full_attention")
        bytes_per_element = torch.finfo(model.dtype).bits // 8
        return (
            num_kv_layers
            * 2
            * text_config.num_key_value_heads
            * text_config.head_dim
            * bytes_per_element
            * batch_size
        )

    def kvzip_apply_query_rope(self, queries: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rotary_dim = cos.shape[-1]
        queries_rotary, queries_pass = queries[..., :rotary_dim], queries[..., rotary_dim:]
        queries_rotary = (queries_rotary * cos) + (rotate_half(queries_rotary) * sin)
        return torch.cat((queries_rotary, queries_pass), dim=-1)

    def kvzip_forward_kwargs(self) -> dict[str, int]:
        return {"logits_to_keep": 1}

    def kvzip_snapshot_reconstruction_state(self, cache):
        return self.snapshot_cache_state(cache)

    def kvzip_restore_reconstruction_state(self, cache, snapshot) -> None:
        if snapshot is not None:
            self.restore_cache_state(cache, snapshot)

    @contextmanager
    def cached_continuation(self, model):
        """Preserve DeltaNet state during cached multi-token continuations.

        The Qwen3.5 implementation in the Transformers version used by this
        project consumes its recurrent/convolution cache only when a DeltaNet
        layer receives one token.  NVIDIA KVPress intentionally forwards a
        complete question or KVzip reconstruction chunk in one model call.
        Keep that upstream call structure intact, but supply each DeltaNet
        chunk kernel with its saved recurrent state and convolution history.
        Full-attention layers and all standard-model behavior are unchanged.
        """
        language_model = self.get_language_model(model)
        text_config = self.get_text_config(model)
        patched_forwards = []

        for layer_idx, layer in enumerate(language_model.layers):
            if text_config.layer_types[layer_idx] != "linear_attention":
                continue

            linear_attention = layer.linear_attn
            original_forward = linear_attention.forward

            def wrapped_forward(
                module_self,
                hidden_states,
                cache_params=None,
                cache_position=None,
                attention_mask=None,
                _original_forward=original_forward,
                **kwargs,
            ):
                # Transformers 5.15+ carries multi-token continuation state
                # in cache.layers[layer_idx] and handles it natively.  The
                # compatibility shim below is only for older flat-list
                # Qwen3.5 caches.
                if cache_params is not None and hasattr(cache_params, "layers"):
                    return _original_forward(
                        hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
                has_previous_state = False
                if cache_params is not None:
                    has_previous_state = getattr(cache_params, "has_previous_state", False)
                    if callable(has_previous_state):
                        has_previous_state = has_previous_state(module_self.layer_idx)

                if not has_previous_state or hidden_states.shape[1] <= 1 or cache_position is None:
                    return _original_forward(
                        hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                        **kwargs,
                    )

                previous_conv_state = cache_params.conv_states[module_self.layer_idx]
                previous_recurrent_state = cache_params.recurrent_states[module_self.layer_idx]
                original_causal_conv1d_fn = module_self.causal_conv1d_fn
                original_chunk_gated_delta_rule = module_self.chunk_gated_delta_rule

                def cached_causal_conv1d_fn(x, weight, bias, activation, seq_idx=None):
                    sequence_length = x.shape[-1]
                    combined_states = torch.cat((previous_conv_state, x), dim=-1)
                    cache_params.conv_states[module_self.layer_idx] = combined_states[
                        :, :, -module_self.conv_kernel_size :
                    ]

                    if original_causal_conv1d_fn is not None:
                        output = original_causal_conv1d_fn(
                            x=combined_states,
                            weight=weight,
                            bias=bias,
                            activation=activation,
                            seq_idx=seq_idx,
                        )
                    else:
                        output = F.silu(module_self.conv1d(combined_states)[:, :, : combined_states.shape[-1]])
                    return output[:, :, -sequence_length:]

                def cached_chunk_gated_delta_rule(*args, **chunk_kwargs):
                    chunk_kwargs["initial_state"] = previous_recurrent_state
                    return original_chunk_gated_delta_rule(*args, **chunk_kwargs)

                # Reuse the installed Qwen3.5 forward and kernels.  These two
                # temporary call-level shims provide the state that its
                # multi-token branch otherwise discards.
                module_self.causal_conv1d_fn = cached_causal_conv1d_fn
                module_self.chunk_gated_delta_rule = cached_chunk_gated_delta_rule
                try:
                    return _original_forward(
                        hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
                finally:
                    module_self.causal_conv1d_fn = original_causal_conv1d_fn
                    module_self.chunk_gated_delta_rule = original_chunk_gated_delta_rule

            linear_attention.forward = MethodType(wrapped_forward, linear_attention)
            patched_forwards.append((linear_attention, original_forward))

        try:
            yield
        finally:
            for linear_attention, original_forward in patched_forwards:
                linear_attention.forward = original_forward

    def get_keys_and_values(self, cache, layer_idx: int):
        if hasattr(cache, "layers"):
            layer = cache.layers[layer_idx]
            if isinstance(cache, QuantizedCache):
                return layer._dequantize(layer._quantized_keys), layer._dequantize(layer._quantized_values)
            return layer.keys, layer.values
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    def set_keys_and_values(self, cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        if hasattr(cache, "layers"):
            layer = cache.layers[layer_idx]
            if isinstance(cache, QuantizedCache):
                layer._quantized_keys = layer._quantize(keys, axis=layer.axis_key)
                layer._quantized_values = layer._quantize(values, axis=layer.axis_value)
                layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                layer.values = torch.zeros(0, dtype=values.dtype, device=values.device)
                layer.cumulative_length = keys.shape[-2]
            else:
                layer.keys = keys
                layer.values = values
            return
        cache.key_cache[layer_idx] = keys
        cache.value_cache[layer_idx] = values

    def cache_seq_lengths(self, cache) -> list[int]:
        if hasattr(cache, "layers"):
            return [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache.layers))]
        return [cache.get_seq_length(layer_idx) for layer_idx in cache.transformer_layers]

    def truncate_cache(self, cache, cache_seq_lengths: list[int]) -> None:
        if hasattr(cache, "layers"):
            for layer_idx, sequence_length in enumerate(cache_seq_lengths):
                layer = cache.layers[layer_idx]
                if hasattr(layer, "keys") and layer.keys is not None:
                    layer.keys = layer.keys[:, :, :sequence_length]
                    layer.values = layer.values[:, :, :sequence_length]
            return
        for layer_idx, sequence_length in zip(cache.transformer_layers, cache_seq_lengths):
            if cache.key_cache[layer_idx] is not None:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :sequence_length]
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :sequence_length]

    def snapshot_cache_state(self, cache):
        if hasattr(cache, "layers"):
            layers = []
            for layer in cache.layers:
                state = {}
                for name in ("keys", "values", "_quantized_keys", "_quantized_values", "cumulative_length"):
                    if hasattr(layer, name):
                        value = getattr(layer, name)
                        state[name] = value.clone() if torch.is_tensor(value) else value
                for name in (
                    "conv_states",
                    "recurrent_states",
                    "has_previous_state",
                    "is_conv_states_initialized",
                    "is_recurrent_states_initialized",
                    "record_past",
                ):
                    if not hasattr(layer, name):
                        continue
                    value = getattr(layer, name)
                    if isinstance(value, dict):
                        state[name] = {
                            key: item.clone() if torch.is_tensor(item) else item
                            for key, item in value.items()
                        }
                    elif torch.is_tensor(value):
                        state[name] = value.clone()
                    else:
                        state[name] = value
                layers.append(state)
            metadata = {
                name: getattr(cache, name)
                for name in ("_seen_tokens", "seen_tokens", "cache_position")
                if hasattr(cache, name)
            }
            return {"layers": layers, "metadata": metadata}

        # Full-attention K/V updates replace list entries with concatenated
        # tensors, so retaining their references is enough and avoids copying
        # a potentially very large context cache. DeltaNet decode can mutate
        # its fixed-size states in place, so those states must be cloned.
        return {
            "key_cache": list(cache.key_cache),
            "value_cache": list(cache.value_cache),
            "recurrent_states": [state.clone() if state is not None else None for state in cache.recurrent_states],
            "conv_states": [state.clone() if state is not None else None for state in cache.conv_states],
            "metadata": {
                name: getattr(cache, name)
                for name in (
                    "layer_types",
                    "transformer_layers",
                    "last_linear_layer",
                    "_seen_tokens",
                    "seen_tokens",
                    "cache_position",
                )
                if hasattr(cache, name)
            },
        }

    def restore_cache_state(self, cache, snapshot):
        if hasattr(cache, "layers") and "layers" in snapshot:
            for layer, state in zip(cache.layers, snapshot["layers"]):
                for name, value in state.items():
                    if isinstance(value, dict):
                        setattr(
                            layer,
                            name,
                            {
                                key: item.clone() if torch.is_tensor(item) else item
                                for key, item in value.items()
                            },
                        )
                    elif torch.is_tensor(value):
                        setattr(layer, name, value.clone())
                    else:
                        setattr(layer, name, value)
            for name, value in snapshot["metadata"].items():
                setattr(cache, name, value)
            return

        cache.key_cache = list(snapshot["key_cache"])
        cache.value_cache = list(snapshot["value_cache"])
        # Clone on every restore so later in-place decode updates cannot alter
        # the master snapshot used by subsequent questions/chunks.
        cache.recurrent_states = [
            state.clone() if state is not None else None for state in snapshot["recurrent_states"]
        ]
        cache.conv_states = [state.clone() if state is not None else None for state in snapshot["conv_states"]]
        for name, value in snapshot["metadata"].items():
            setattr(cache, name, value)


def get_model_adapter(model) -> ModelAdapter:
    if model.config.model_type in ("qwen3_5", "qwen3_5_text"):
        return Qwen35ModelAdapter()
    return StandardModelAdapter()
