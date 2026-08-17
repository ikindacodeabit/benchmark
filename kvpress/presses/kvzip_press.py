# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Generator, List

import torch
from torch import nn
# ORIGINAL: from transformers import AutoTokenizer, Gemma3PreTrainedModel, PreTrainedModel, PreTrainedTokenizer, QuantizedCache
# QWEN3.5 CHANGE: Cache reads/writes are delegated to the model adapter.
from transformers import AutoTokenizer, Gemma3PreTrainedModel, PreTrainedModel, PreTrainedTokenizer
from transformers.models.llama.modeling_llama import rotate_half

# QWEN3.5 CHANGE: Qwen3.5 has hybrid full-attention and linear-attention layers.
from kvpress.model_adapter import get_model_adapter
from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress
# ORIGINAL: from kvpress.utils import extract_keys_and_values, get_prerope_query_states
from kvpress.utils import get_prerope_query_states

logger = logging.getLogger(__name__)


@dataclass
class KVzipPress(BasePress):
    """
    KVzip identifies the importance of KV pairs through context reconstruction,
    enabling effective query-agnostic KV cache compression.

    In this code, we implement KVzip with minimal changes to this repository.
    For a fully optimized implementation with actual compression,
    please refer to the original repository,
    which also provides a version without runtime compression overhead (at the cost of performance).
    Original repository (https://github.com/snu-mllab/KVzip).

    Based on KVzip (https://arxiv.org/abs/2505.23416).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers.
        When False, while the overall KV cache compression ratio is maintained,
        each layer has a different compression ratio.
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks.
    kvzip_plus_normalization: bool, default=False
        Whether to enable KVzip+ normalization.
    """

    compression_ratio: float = 0.0
    layerwise: bool = False
    n_sink: int = 4
    kvzip_plus_normalization: bool = False

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        logger.warning(
            "KVzipPress requires multiple forward passes for chunked context reconstruction, "
            "resulting in a computational overhead of 2–3 times the initial prefilling cost. "
            "This significantly increases the overall prefilling time compared to other compression methods, "
            "which is inherent to the KVzip algorithm design."
        )
        self._reset_internal_parameters()

    def _reset_internal_parameters(self):
        self.context_length = 0
        self.prefix_length = 0

        self._suffix_ids = None
        self._context_ids = None
        self._cache = None

        self.score_val = None
        self.causal_mask_score = None
        self.start_idx = 0
        self.end_idx = 0

        # QWEN3.5 CHANGE: Map physical full-attention layers to dense KVzip score rows.
        self._model_adapter = None
        self._attention_layer_indices = []
        self._score_layer_position = {}

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager that handles both initial prefilling and KVzip scoring/compression.

        This overrides the base class __call__ method to implement the full KVzip algorithm:
        1. First yield: allows initial prefilling with context
        2. After yield: performs KVzip scoring and compression using context reconstruction
        """
        # ORIGINAL: if not isinstance(model, SUPPORTED_MODELS):
        # QWEN3.5 CHANGE: Qwen3.5 is supported through its adapter.
        adapter = get_model_adapter(model)
        if not isinstance(model, SUPPORTED_MODELS) and not adapter.is_qwen35:
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")

        if isinstance(model, Gemma3PreTrainedModel):
            raise ValueError("KVzipPress is not supported for Gemma3ForCausalLM")

        self.post_init_from_model(model)

        # Store model reference for later use
        # QWEN3.5 CHANGE: Only full-attention layers receive KVzip hooks/scores.
        self._model_adapter = adapter
        self._attention_layer_indices = [idx for idx, _ in adapter.iter_kv_attention_layers(model)]
        self._score_layer_position = {
            layer_idx: position for position, layer_idx in enumerate(self._attention_layer_indices)
        }
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)

        # Get suffix_ids directly using tokenizer's chat template (do this once, not in hook)
        if tokenizer.chat_template is None:
            prefix_text = ""
            suffix_text = "\n"  # Default suffix for models without chat template
        else:
            # Use a dummy context to extract the question suffix from chat template
            dummy_context = "dummy context"
            separator = "\n" + "#" * len(dummy_context)
            temp_context = tokenizer.apply_chat_template(
                [{"role": "user", "content": dummy_context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            context, suffix_text = temp_context.split(separator)
            prefix_text = context.split(dummy_context)[0]

        # Tokenize suffix directly to ids
        self.prefix_length = tokenizer.encode(prefix_text, return_tensors="pt", add_special_tokens=False).shape[-1]
        self._suffix_ids = tokenizer.encode(suffix_text, return_tensors="pt", add_special_tokens=False)

        # Register hook to store the pointer for past_key_values
        original_forward = model.model.forward

        def wrapped_forward(model_self, *args, **kwargs):
            self._context_ids = kwargs["input_ids"]
            self._cache = kwargs["past_key_values"]
            return original_forward(*args, **kwargs)

        model.model.forward = MethodType(wrapped_forward, model.model)

        hooks = []
        try:
            try:
                yield
            finally:
                model.model.forward = original_forward

            # After yield: KVzip scoring and compression phase
            if self.compression_ratio > 0 and self._context_ids is not None:
                # Now register attention hooks for compression
                # ORIGINAL:
                # for layer in model.model.layers:
                #     layer.self_attn.rotary_emb = model.model.rotary_emb
                #     hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
                # QWEN3.5 CHANGE: Skip DeltaNet layers, which have no growing K/V cache.
                language_model = adapter.get_language_model(model)
                for layer_idx, attention in adapter.iter_kv_attention_layers(model):
                    attention.rotary_emb = language_model.rotary_emb
                    attention.layer_idx = layer_idx
                    hooks.append(attention.register_forward_hook(self.forward_hook, with_kwargs=True))

                self._perform_kvzip_compression(model, tokenizer)
        finally:
            for hook in hooks:
                hook.remove()
            self._reset_internal_parameters()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Override the forward_hook of BasePress.
        During the forward_hook, KVzip only calculates importance scores,
        aggregates scores across all layers, and then performs compression.
        """

        hidden_states = kwargs["hidden_states"]
        # ORIGINAL: cache = kwargs.get("past_key_values", None) or kwargs.get("past_key_value", None)
        # ORIGINAL: cache_layer = cache.layers[module.layer_idx]
        # QWEN3.5 CHANGE: The installed Qwen3.5 cache exposes top-level cache lists.
        cache = kwargs.get("past_key_values", None)
        if cache is None:
            cache = kwargs.get("past_key_value", None)
        adapter = self._model_adapter

        # ORIGINAL: keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = adapter.get_keys_and_values(cache, module.layer_idx)

        # Compute importance scores for KV pairs in the prefilled context,
        # retaining only the originally prefilled KV pairs.
        keys, values = self.score_kvzip(module, hidden_states, keys, values, output[1], kwargs)

        # ORIGINAL:
        # if isinstance(cache, QuantizedCache):
        #     cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
        #     cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
        #     cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        #     cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        #     cache_layer.cumulative_length = keys.shape[2]
        # else:
        #     cache_layer.keys = keys
        #     cache_layer.values = values
        # QWEN3.5 CHANGE: Adapter preserves the original standard/quantized behavior
        # and handles Qwen3.5's top-level key_cache/value_cache layout.
        adapter.set_keys_and_values(cache, module.layer_idx, keys, values)

        return output

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Perform the KVzip scoring and compression algorithm.
        """

        # Prepare chunked inputs for context reconstruction
        self.context_length = self._context_ids.shape[1]
        chunked_context_pairs = self.prepare(model, tokenizer)

        # Perform scoring through context reconstruction
        # Use the stored cache from the initial forward pass
        self.start_idx = self.prefix_length
        # ORIGINAL:
        # for prefill_ids, repeat_ids in chunked_context_pairs:
        #     self.end_idx = self.start_idx + prefill_ids.shape[1]
        #     model(
        #         input_ids=repeat_ids.to(model.device),
        #         past_key_values=self._cache,
        #         num_logits_to_keep=1,
        #     )
        #     self.start_idx = self.end_idx
        # QWEN3.5 CHANGE: Adapter operations are no-ops for standard models,
        # preserving the original Qwen3 reconstruction behavior exactly.
        context_snapshot = self._model_adapter.kvzip_snapshot_reconstruction_state(self._cache)
        try:
            # Keep NVIDIA KVPress's reconstruction chunks unchanged.  On
            # Qwen3.5, only the DeltaNet layers receive their restored context
            # state when processing each chunk.
            with self._model_adapter.cached_continuation(model):
                for prefill_ids, repeat_ids in chunked_context_pairs:
                    self.end_idx = self.start_idx + prefill_ids.shape[1]
                    self._model_adapter.kvzip_restore_reconstruction_state(self._cache, context_snapshot)
                    model(
                        input_ids=repeat_ids.to(model.device),
                        past_key_values=self._cache,
                        **self._model_adapter.kvzip_forward_kwargs(),
                    )
                    self.start_idx = self.end_idx
        finally:
            self._model_adapter.kvzip_restore_reconstruction_state(self._cache, context_snapshot)

        # Perform final compression
        self.compress_post(model)

    def _chunk_fn(self, ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """
        Chunk input tokens
        """
        ctx_len = ctx_ids.shape[1]
        if ctx_len > chunk_size:
            chunk_num = (ctx_len - 1) // chunk_size + 1

            chunked_input_ids = []
            for i in range(chunk_num):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                a_ids = ctx_ids[:, start:end]
                if a_ids.shape[1] == 0:
                    continue
                chunked_input_ids.append(a_ids)
        else:
            chunked_input_ids = [ctx_ids]

        return chunked_input_ids

    def prepare(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 2048,
        prev_postfix_size=8,
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare chunked inputs for KV importance scoring with context reconstruction
        """
        ctx_ids = self._context_ids[:, self.prefix_length :].to("cpu")

        # initialize score values
        # ORIGINAL: model.config.num_hidden_layers,
        # ORIGINAL: model.config.num_key_value_heads,
        # QWEN3.5 CHANGE: Score only full-attention layers and read the text config.
        text_config = self._model_adapter.get_text_config(model)
        self.score_val = torch.zeros(
            (
                len(self._attention_layer_indices),
                1,
                text_config.num_key_value_heads,
                self.context_length,
            ),  # only support batch size of 1
            dtype=model.dtype,
            device=model.device,
        )
        self.score_val[..., : self.n_sink] = 1.0

        chunked_context_pairs = []
        chunked_input_ids = self._chunk_fn(ctx_ids, chunk_size)
        for i, a_ids in enumerate(chunked_input_ids):
            if i == 0:
                prompt = "\n\nRepeat the previous context exactly."
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                prompt = "\n\nRepeat the part of the previous context exactly, starting with"
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
                postfix_prev = chunked_input_ids[i - 1][:, -prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            chunked_context_pairs.append((a_ids, torch.cat([q_ids, self._suffix_ids, a_ids], dim=1)))

        return chunked_context_pairs

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """
        Define causal mask shared across layers
        """
        mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """
        Apply causal masking
        """
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        self.causal_mask_score = self.causal_mask_score.to(attn_weights.device)
        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score

    def score_kvzip(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the maximum cross-attention scores during context reconstruction,
        and return slices of the keys and values containing only the originally prefilled KV pairs,
        i.e., excluding KV pairs from repeated contexts.
        The computed scores are stored in self.score_val.
        """

        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        num_heads_kv = module.config.num_key_value_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // num_heads_kv

        queries = get_prerope_query_states(module, hidden_states)

        # Apply RoPE
        cos, sin = kwargs["position_embeddings"]
        # ORIGINAL: queries = (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))
        # QWEN3.5 CHANGE: The standard adapter executes the original line;
        # Qwen3.5's adapter handles only its partial-RoPE dimensions.
        queries = self._model_adapter.kvzip_apply_query_rope(queries, cos, sin)
        queries = queries.view(bsz, num_heads_kv, num_key_value_groups, q_len, head_dim)

        # Subsample keys
        sink = min(self.n_sink, self.start_idx)
        ctx_len = self.end_idx - self.start_idx
        keys_subsampled = torch.cat(
            [
                keys[:, :, :sink],  # attention sink tokens (generally system prompt)
                keys[:, :, self.start_idx : self.end_idx],  # KV chunk in the cache
                keys[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2,
        )
        keys_subsampled = keys_subsampled.unsqueeze(2).transpose(-2, -1).contiguous()

        # Compute attention
        attn_weights = torch.matmul(queries, keys_subsampled) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if self.kvzip_plus_normalization:
            # Divide by ||h|| (by row)
            h_norm = torch.norm(hidden_states, dim=-1)
            attn_weights = torch.einsum("b h g t i, b t -> b h g t i", attn_weights, 1 / h_norm)

            # Multiply by ||WoV|| (by column)
            Wo = module.o_proj.weight.transpose(0, 1)
            Wo = Wo.view(num_heads_kv, num_key_value_groups, module.head_dim, module.config.hidden_size)
            values_subsampled = torch.cat(
                [values[:, :, :sink], values[:, :, self.start_idx : self.end_idx], values[:, :, -q_len:]], dim=2
            )
            values_subsampled = values_subsampled.unsqueeze(2).transpose(-2, -1).contiguous()
            V = values_subsampled.repeat_interleave(module.num_key_value_groups, axis=2)
            WoV_norm = torch.einsum("h g i j, b h g i t -> b h g t j", Wo, V).norm(dim=-1)
            attn_weights = torch.einsum("b h g i t, b h g t -> b h g i t", attn_weights, WoV_norm)

        attn_weights = attn_weights[..., sink : sink + ctx_len]
        scores = attn_weights.amax(dim=(-3, -2))  # max over group, q

        # ORIGINAL: layer_idx = int(module.layer_idx)
        # ORIGINAL: self.score_val[layer_idx][..., self.start_idx : self.end_idx] = scores
        # QWEN3.5 CHANGE: Physical layer indices are sparse; score rows are dense.
        layer_position = self._score_layer_position[int(module.layer_idx)]
        self.score_val[layer_position][..., self.start_idx : self.end_idx] = scores  # update score

        # Retain the originally prefilled context KV pairs and exclude KV pairs from the repeated context
        keys, values = keys[:, :, : self.context_length], values[:, :, : self.context_length]
        return keys, values

    def compress_post(self, model: PreTrainedModel):
        """
        Obtain the indices of KV pairs to be evicted.
        Adopted from adakv_press.compress (fake compression). KVzip does not rely on safeguards.
        """
        if self.compression_ratio > 0:
            n_layer, bsz, num_key_value_heads, ctx_len = self.score_val.shape

            # calculate the pruned KV pairs across layers
            if self.layerwise:
                nl = int(bsz * num_key_value_heads * ctx_len * self.compression_ratio)
                n_pruned_layers = nl * torch.ones(n_layer, device=self.score_val.device, dtype=torch.int)
            else:
                n_pruned_indices = int(self.score_val.numel() * self.compression_ratio)
                pruned_indices = torch.topk(-self.score_val.reshape(-1), n_pruned_indices).indices
                n_tokens_per_layer = bsz * num_key_value_heads * ctx_len
                n_pruned_layers = torch.bincount(pruned_indices // n_tokens_per_layer, minlength=n_layer).int()

            # ORIGINAL:
            # for layer in model.model.layers:
            #     module = layer.self_attn
            #     layer_idx = int(module.layer_idx)
            # QWEN3.5 CHANGE: Iterate only full-attention layers.
            adapter = self._model_adapter
            for layer_idx, module in adapter.iter_kv_attention_layers(model):
                layer_idx = int(layer_idx)

                assert module.config._attn_implementation != "eager", "eager mode not supported"

                # ORIGINAL: scores = self.score_val[layer_idx]
                scores = self.score_val[self._score_layer_position[layer_idx]]

                # Compute bottom-k across heads
                # ORIGINAL: n_pruned = n_pruned_layers[layer_idx].cpu()
                layer_position = self._score_layer_position[layer_idx]
                n_pruned = n_pruned_layers[layer_position].cpu()
                indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()

                # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for details
                batch_indices = torch.arange(bsz, device=n_pruned.device).repeat_interleave(n_pruned)
                head_indices = indices // ctx_len
                seq_indices = indices % ctx_len
                module.masked_key_indices = (batch_indices, head_indices, seq_indices)
