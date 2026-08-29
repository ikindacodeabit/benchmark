# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
from torch import nn
from transformers import (
    Gemma3ForConditionalGeneration,
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    PreTrainedModel,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
)

from kvpress.model_adapter import get_model_adapter

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = (
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Gemma3ForConditionalGeneration,
)


def _bind_model_adapter(press, adapter, model, _seen=None) -> None:
    """Give ``press`` and every press it delegates to the model adapter.

    Wrapper presses (ComposedPress, PerLayerCompressionPress,
    PrefillDecodingPress, AdaKVPress, ...) call the INNER press's
    forward_hook directly, so the inner press needs the adapter even though
    its own ``__call__`` never runs. Without this the inner press raised
    ``AttributeError: no attribute '_model_for_adapter'`` on the first
    compressed layer. Inner presses are found by walking attributes rather
    than a fixed name list, because wrappers disagree on the field name
    (press / presses / prefilling_press / decoding_press).
    """
    if _seen is None:
        _seen = set()
    if id(press) in _seen:
        return
    _seen.add(id(press))

    press._model_adapter = adapter
    press._model_for_adapter = model
    for value in list(vars(press).values()):
        for candidate in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(candidate, BasePress):
                _bind_model_adapter(candidate, adapter, model, _seen)


def _supported_on_qwen35(press) -> bool:
    """Whether a press is safe on Qwen3.5's packed query projection.

    Only presses that score from keys alone are: Qwen3.5 packs the attention
    gate into q_proj, so anything reading query states needs the unpacking in
    utils.get_prerope_query_states.

    Checked by class rather than by class NAME, and through wrappers. The name
    check this replaces was wrong in both directions: a ComposedPress or
    AdaKVPress around KnormPress was rejected despite reading no queries, and
    any subclass that happened to be named KnormPress was accepted.
    """
    # Imported here, not at module scope: these presses import base_press.
    from kvpress.presses.knorm_press import KnormPress
    from kvpress.presses.random_press import RandomPress

    inner = getattr(press, "press", None)
    if inner is not None:
        return _supported_on_qwen35(inner)
    nested = getattr(press, "presses", None)
    if nested is not None:
        return all(_supported_on_qwen35(p) for p in nested)
    return isinstance(press, (RandomPress, KnormPress))


@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.

    This class provides the foundation for implementing various key-value cache compression
    techniques. Subclasses must implement the `compress` method to define their specific
    compression logic.

    The compression is applied only during pre-filling (not during generation).
    """

    def post_init_from_model(self, model: PreTrainedModel):
        """
        Optional method to initialize press parameters from the model
        """
        pass

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core logic of the compression method.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where compression is applied.
        hidden_states : torch.Tensor
            Hidden states of the current layer with shape (batch_size, seq_len, hidden_dim).
            These represent the input to the attention layer.
        keys : torch.Tensor
            Key tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are keys ready for compression.
        values : torch.Tensor
            Value tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are values ready for compression.
        attentions : torch.Tensor
            Attention weights from the layer with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if attention weights are not computed or needed.
        kwargs : dict
            Additional keyword arguments from the forward pass.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the compressed keys and values tensors. The returned tensors
            should have reduced sequence length dimension compared to the input tensors.
        """

        raise NotImplementedError("compress method must be implemented in subclass")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.

        This hook automatically applies compression during the pre-filling phase by:
        1. Checking if we're still in pre-filling (not generation) phase
        2. Extracting keys and values from the cache (handling quantization)
        3. Calling the compress method to reduce the cache size
        4. Updating the cache with compressed keys and values

        The hook ensures compression is only applied during pre-filling and correctly
        handles both quantized and unquantized caches.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        input : list[torch.Tensor]
            Input tensors to the forward pass of the attention layer. This parameter
            is provided by PyTorch's hook mechanism but not used in the default implementation.
        kwargs : dict
            Keyword arguments passed to the attention layer's forward method, including:
            - hidden_states: Input embeddings to the attention layer
            - past_key_values: The KV cache object being modified
            - cache_position: Position indices indicating where we are in the sequence
            - position_embeddings: RoPE embeddings if applicable
        output : list
            Output from the attention layer's forward pass. Contains:
            - [0]: Hidden states output
            - [1]: Attention weights (may be None)

        Returns
        -------
        list
            The potentially modified output from the forward pass. This
            is the same as the input output, but the underlying cache has been compressed in-place.
        """

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        adapter = getattr(self, "_model_adapter", None)
        if adapter is None:
            # Only __call__ sets this, and it sets both fields together -- so a
            # missing adapter meant the old fallback dereferenced an equally
            # missing _model_for_adapter and raised an opaque AttributeError.
            raise RuntimeError(
                f"{type(self).__name__}.forward_hook was called outside the press context manager. "
                "Register hooks via `with press(model):` so the model adapter is set up."
            )
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling
        if kwargs["cache_position"][-1] > q_len:
            return output

        keys, values = adapter.get_keys_and_values(cache, module.layer_idx)

        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        adapter.set_keys_and_values(cache, module.layer_idx, keys, values)

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.

        This method registers forward hooks on all attention layers of the model to enable
        automatic KV cache compression during the pre-filling phase. The hooks are automatically
        removed when exiting the context manager.

        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            The transformer model to apply compression to.

        Examples
        --------
        >>> from kvpress import KnormPress
        >>> press = KnormPress(compression_ratio=0.5)
        >>> with press(model):
        ...     # Forward pass with compression applied
        ...     outputs = model(input_ids, past_key_values=cache)
        """
        adapter = get_model_adapter(model)
        if adapter.is_qwen35 and not _supported_on_qwen35(self):
            raise NotImplementedError(
                "Qwen3.5 currently supports only no_press, random, and knorm; "
                "query-aware presses need Qwen3.5 query-projection handling."
            )
        if not isinstance(model, SUPPORTED_MODELS) and not adapter.is_qwen35:
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")

        if isinstance(model, Gemma3ForConditionalGeneration):
            logger.warning_once("Compression in Gemma3 is only applied to layer without sliding window attention")

        self.post_init_from_model(model)
        hooks = []
        _bind_model_adapter(self, adapter, model)
        try:
            language_model = adapter.get_language_model(model)
            for layer_idx, attention in adapter.iter_kv_attention_layers(model):
                if isinstance(model, Gemma3ForConditionalGeneration) and attention.is_sliding:
                    # Skip layers with sliding window attention, only for Gemma3
                    continue
                attention.rotary_emb = language_model.rotary_emb
                attention.layer_idx = layer_idx
                hooks.append(attention.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
            # Presses in evaluation's registry are module-level singletons, so
            # holding the model here kept it alive for the process lifetime and
            # left a stale adapter for the next model to pick up.
            _bind_model_adapter(self, None, None)
