# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import contextlib
import os
import logging
from typing import Any, Callable, Optional

import torch
from transformers import AutoModelForCausalLM, Cache, DynamicCache, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from kvpress.presses.base_press import BasePress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.dms_press import DMSPress
from kvpress.presses.finch_press import FinchPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.prefill_decoding_press import PrefillDecodingPress
from kvpress.model_adapter import get_model_adapter

logger = logging.getLogger(__name__)

MEMORY_UNIT_TO_BYTES = {
    "MB": 1000**2,
    "GB": 1000**3,
}


class KVPressTextGenerationPipeline(Pipeline):
    """
    Pipeline for key-value cache compression in causal language models.

    Enables efficient processing of long contexts by applying KV cache compression
    during pre-filling, then generating answers using greedy decoding.

    Example:
    ```python
    pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    press = SnapKVPress(compression_ratio=0.5)
    result = pipeline(context="Long text...", question="A question about the long context.", press=press)
    ```
    """

    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        enable_thinking: bool = False,
        cache: Optional[Cache] = None,
        memory_budget: Optional[float] = None,
        memory_budget_unit: str = "GB",
        question_progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs,
    ):
        """
        Sanitize the input parameters for the pipeline.
        The user can either provide a single question or a list of questions to be asked about the context.

        Parameters
        ----------
        question : str, optional
            The question to be asked about the context. Exclusive with `questions`.
        questions : list[str], optional
            A list of questions to be asked about the context. Exclusive with `question`.
        answer_prefix : str, optional
            The prefix to be added to the generated answer.
        press : BasePress, optional
            The key-value cache compression method to apply during pre-filling.

            Accepts any KVPress compression method (SnapKVPress, KnormPress,
            ExpectedAttentionPress, BlockPress, AdaKVPress, ComposedPress, etc.).
            If None, no compression is applied.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate for each answer.
        max_context_length : int, optional
            The maximum number of tokens in the context. By default will use the maximum length supported by the model.
        enable_thinking: bool = False,
            Whether to enable thinking in the chat template (chat template must support this argument)
        cache : Cache, optional
            The cache to use for the forward pass. Defaults to None (DynamicCache).
        **kwargs : dict
            Additional keyword arguments, currently ignored.

        Returns
        -------
        Tuple[dict, dict, dict]
            A tuple containing three dictionaries:
                - preprocess_kwargs: The keyword arguments for the preprocess function.
                - forward_kwargs: The keyword arguments for the forward function.
                - postprocess_kwargs: The keyword arguments for the postprocess function.
        """

        answer_prefix = answer_prefix or ""
        postprocess_kwargs = {"single_question": questions is None}
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        questions = questions or ([question] if question else [""])
        requested_max_context_length = max_context_length

        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))  # 1e10 to avoid overflow
        logger.info(
            "Context length limit: requested=%s, effective=%d, tokenizer_model_max_length=%s",
            requested_max_context_length,
            max_context_length,
            self.tokenizer.model_max_length,
        )
        preprocess_kwargs = {
            "questions": questions,
            "answer_prefix": answer_prefix,
            "max_context_length": max_context_length,
            "enable_thinking": enable_thinking,
        }
        forward_kwargs = {
            "press": press,
            "max_new_tokens": max_new_tokens,
            "memory_budget": memory_budget,
            "memory_budget_unit": memory_budget_unit,
            "cache": cache,
            "question_progress_callback": question_progress_callback,
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def _compute_kv_bytes_per_token(self, batch_size: int = 1) -> int:
        """Return the number of bytes used by one token across the model's KV cache."""
        return get_model_adapter(self.model).kv_bytes_per_token(self.model, batch_size)

    def _compute_token_budget_from_memory(
        self, memory_budget: float, memory_budget_unit: str = "GB", batch_size: int = 1
    ) -> tuple[int, int, int]:
        """
        Compute how many tokens can fit in the KV cache for an explicit memory unit.

        Returns the token budget, bytes per token, and budget in bytes.
        """
        if memory_budget <= 0:
            raise ValueError(f"memory_budget must be positive, got {memory_budget}")

        normalized_unit = memory_budget_unit.strip().upper()
        if normalized_unit not in MEMORY_UNIT_TO_BYTES:
            supported_units = ", ".join(MEMORY_UNIT_TO_BYTES)
            raise ValueError(
                f"Unsupported memory_budget_unit={memory_budget_unit!r}. Supported units: {supported_units}"
            )

        bytes_per_token = self._compute_kv_bytes_per_token(batch_size)
        memory_budget_bytes = int(memory_budget * MEMORY_UNIT_TO_BYTES[normalized_unit])
        token_budget = int(memory_budget_bytes // bytes_per_token)
        if token_budget < 1:
            raise ValueError(
                f"A {memory_budget:g} {memory_budget_unit} KV budget is smaller than one KV token "
                f"({bytes_per_token} bytes per token for this model)."
            )

        return token_budget, bytes_per_token, memory_budget_bytes

    @staticmethod
    def _compute_context_compression_ratio(context_length: int, token_budget: int) -> tuple[int, float]:
        """Return retained context tokens and the compression ratio for a token budget."""
        if context_length < 1:
            raise ValueError(f"context_length must be positive, got {context_length}")
        if token_budget < 1:
            raise ValueError(f"token_budget must be positive, got {token_budget}")

        retained_context_tokens = min(context_length, token_budget)
        compression_ratio = 1 - (retained_context_tokens / context_length)
        return retained_context_tokens, compression_ratio

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
        enable_thinking: bool = False,
    ):
        """
        Apply chat template and tokenize the context and questions.

        Prepares input text for KV cache compression and generation by applying
        appropriate chat templates and tokenizing. Handles models with and without
        chat templates.

        Parameters
        ----------
        context : str
            Long context text to be compressed using the press method.
        questions : list[str]
            Questions to be asked about the context.
        answer_prefix : str
            Optional prefix for generated answers.
        max_context_length : int
            Maximum tokens allowed in context (truncated if exceeded).
        enable_thinking : bool
            Whether to enable thinking in the chat template (chat template must support this argument)

        Returns
        -------
        dict[str, GenericTensor]
            Dictionary with "context_ids" and "questions_ids" tensors.
        """

        # Apply chat template if available
        if self.tokenizer.chat_template is None:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context = bos_token + context
            question_suffix = "\n"  # to separate the question from the answer
        else:
            separator = "#" * (len(context) + 10)
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
            context, question_suffix = context.split(separator)

        # Add question_suffix and answer prefix
        # e.g. for llama3.1, question_suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        questions = [question + question_suffix + answer_prefix for question in questions]

        # Tokenize the context and questions
        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) for question in questions
        ]

        original_context_length = context_ids.shape[1]
        # Truncate context
        if context_ids.shape[1] > max_context_length:
            logger.warning(
                f"Context length has been truncated from {context_ids.shape[1]} to {max_context_length} tokens."
            )
            context_ids = context_ids[:, :max_context_length]
        logger.info(
            "Prepared context tokens: original=%d, final=%d, effective_max_context_length=%d",
            original_context_length,
            context_ids.shape[1],
            max_context_length,
        )

        return {"context_ids": context_ids, "questions_ids": question_ids}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
        memory_budget: Optional[float] = None,
        memory_budget_unit: str = "GB",
        question_progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Execute KV cache compression and text generation pipeline.

        Performs context compression using the press method during pre-filling,
        then generates answers using greedy decoding.

        Parameters
        ----------
        input_tensors : dict[str, GenericTensor]
            Tokenized inputs with "context_ids" and "questions_ids".
        max_new_tokens : int, default=50
            Maximum tokens to generate for each answer.
        press : BasePress, optional
            Compression method for context pre-filling. If None, no compression.
        cache : Cache, optional
            Cache object for forward pass. If None, creates new DynamicCache.

        Returns
        -------
        list[str]
            Generated answers for each input question.
        """
        if isinstance(press, (DecodingPress, PrefillDecodingPress)) and len(input_tensors["questions_ids"]) > 1:
            raise ValueError(
                "DecodingPress is not compatible with multiple questions. Please specify a single question."
            )

        context_ids = input_tensors["context_ids"].to(self.model.device)
        # print("Context (raw) :>", self.tokenizer.decode(context_ids[0], skip_special_tokens=False))

        context_length = context_ids.shape[1]

        # Prefilling using the press on the context
        adapter = get_model_adapter(self.model)
        if cache is None:
            cache = DynamicCache()
        
        bytes_per_token = self._compute_kv_bytes_per_token()
        compression_ratio = float(getattr(press, "compression_ratio", 0.0)) if press is not None else 0.0

        retained_context_tokens = max(1, int(context_length * (1 - compression_ratio)))
        budget_stats: dict[str, Any] = {}

        # Dynamically compute the context compression ratio from the KV-cache budget.
        if memory_budget is not None:
            if press is None or not hasattr(press, "compression_ratio"):
                raise ValueError("memory_budget requires a press with a compression_ratio attribute")

            token_budget, bytes_per_token, memory_budget_bytes = self._compute_token_budget_from_memory(
                memory_budget, memory_budget_unit
            )
            retained_context_tokens, ratio = self._compute_context_compression_ratio(context_length, token_budget)
            press.compression_ratio = ratio
            compression_ratio = ratio
            budget_stats = {
                "memory_budget": memory_budget,
                "memory_budget_unit": memory_budget_unit.upper(),
                "token_budget": token_budget,
            }
            logger.info(
                f"memory_budget={memory_budget:g}{memory_budget_unit} -> token_budget={token_budget}, "
                f"context_length={context_length}, retained_context_tokens={retained_context_tokens}, "
                f"compression_ratio={ratio:.6f}"
            )

        retained_kv_bytes = retained_context_tokens * bytes_per_token
        uncompressed_kv_bytes = context_length * bytes_per_token
        self.last_memory_budget_stats = {
            **budget_stats,
            "kv_memory_per_token_kb": bytes_per_token / 1000,
            "context_tokens": context_length,
            "retained_context_tokens": retained_context_tokens,
            "compression_ratio": compression_ratio,
            "retained_kv_memory_mb": retained_kv_bytes / 1000**2,
            "retained_kv_memory_gb": retained_kv_bytes / 1000**3,
            "uncompressed_kv_memory_mb": uncompressed_kv_bytes / 1000**2,
            "uncompressed_kv_memory_gb": uncompressed_kv_bytes / 1000**3,
        }
        # We only perform prefill compression if the press is a prefill press
        perform_prefill_compression = press is not None and not isinstance(press, DecodingPress)
        with press(self.model) if perform_prefill_compression else contextlib.nullcontext():
            # We run the model without the lm head for pre-filling.
            self.model.model(
                input_ids=context_ids,
                past_key_values=cache,
            )
        # We only perform decoding compression if the press is a decoding or prefill decoding press
        perform_decoding_compression = press is not None and isinstance(press, (DecodingPress, PrefillDecodingPress))
        if isinstance(press, DMSPress):
            perform_decoding_compression = press.decoding
        with press(self.model) if perform_decoding_compression else contextlib.nullcontext():
            # Greedy decoding for each question
            answers = []
            context_snapshot = adapter.snapshot_cache_state(cache)
            total_questions = len(input_tensors["questions_ids"])
            for question_number, question_ids in enumerate(input_tensors["questions_ids"], start=1):
                if isinstance(press, KeyRerotationPress) or (isinstance(press, FinchPress) and press.rerotate_keys):
                    context_length = cache.get_seq_length()

                answer = self.generate_answer(
                    question_ids=question_ids.to(self.model.device),
                    cache=cache,
                    context_length=context_length,
                    max_new_tokens=max_new_tokens,
                )
                adapter.restore_cache_state(cache, context_snapshot)

                answers.append(answer)
                if question_progress_callback is not None:
                    question_progress_callback(question_number, total_questions, answer)
        return answers

    def _remove_answer_from_cache(self, cache: Cache, cache_seq_lengths: list[int]):
        # Kept as a compatibility helper for callers outside this pipeline.
        get_model_adapter(self.model).truncate_cache(cache, cache_seq_lengths)

    def generate_answer(
        self,
        question_ids: torch.Tensor,
        cache: Cache,
        context_length: int,
        max_new_tokens: int,
    ) -> str:
        """
        Generate an answer to a question using greedy decoding.

        Parameters
        ----------
        question_ids : torch.Tensor
            The tokenized question.
        cache : Cache
            The compressed key-value cache.
        context_length : int
            The length of the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.
        Returns
        -------
        str
            The generated answer.
        """
        position_ids = torch.arange(
            context_length, context_length + question_ids.shape[1], device=self.model.device
        ).unsqueeze(0)
        debug_generation = os.environ.get("KV_PRESS_DEBUG_GENERATION") == "1"
        if debug_generation:
            cache_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
            logger.info(
                "GEN_DEBUG before_question original_context_length=%d question_tokens=%d "
                "cache_length_min=%d cache_length_max=%d position_start=%d position_end=%d",
                context_length,
                question_ids.shape[1],
                min(cache_lengths),
                max(cache_lengths),
                position_ids[0, 0].item(),
                position_ids[0, -1].item(),
            )

        # if the user doesn't provide a question, skip forward pass
        adapter = get_model_adapter(self.model)
        # NVIDIA KVPress forwards the complete question in one cached model
        # call.  The standard adapter preserves that call exactly; Qwen3.5
        # makes only its DeltaNet layers retain state across that continuation.
        with adapter.cached_continuation(self.model):
            outputs = self.model(
                input_ids=question_ids.to(self.model.device),
                past_key_values=cache,
                position_ids=position_ids,
                **adapter.kvzip_forward_kwargs(),
            )

        if debug_generation:
            logits = outputs.logits[0, -1].float()
            top_values, top_ids = torch.topk(logits, k=5)
            top_tokens = [self.tokenizer.decode([token_id.item()]) for token_id in top_ids]
            logger.info(
                "GEN_DEBUG first_logits finite=%s min=%.6f max=%.6f "
                "top_ids=%s top_tokens=%r top_logits=%s",
                bool(torch.isfinite(logits).all()),
                logits.min().item(),
                logits.max().item(),
                top_ids.tolist(),
                top_tokens,
                [round(value.item(), 6) for value in top_values],
            )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            outputs = self.model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break
        answer = str(self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True))
        return answer

    def postprocess(self, model_outputs, single_question):
        if single_question:
            return {"answer": model_outputs[0]}
        return {"answers": model_outputs}


PIPELINE_REGISTRY.register_pipeline(
    "kv-press-text-generation",
    pipeline_class=KVPressTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
