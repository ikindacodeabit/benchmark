#!/usr/bin/env python3
"""Short GPU smoke test for Qwen3.5 + KVzipPress.

This deliberately uses a tiny context. It validates the adapter-aware KVzip
hook/reconstruction path before launching a long-context benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return a scale-independent error for a smoke-test comparison."""
    reference = reference.float()
    candidate = candidate.float()
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-6)
    return (torch.linalg.vector_norm(reference - candidate) / denominator).item()


def verify_delta_multitoken_continuation(model, tokenizer, adapter, seed_ids: torch.Tensor) -> None:
    """Compare the adapter's multi-token path with native one-token decoding."""
    context_tokens = 64
    repeats = (context_tokens // max(seed_ids.shape[1], 1)) + 1
    context_ids = seed_ids.repeat(1, repeats)[:, :context_tokens].to(model.device)
    continuation_ids = tokenizer(
        " What important fact does this context state?",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)
    if continuation_ids.shape[1] <= 1:
        raise AssertionError("DeltaNet continuation smoke test requires multiple tokens")

    tokenwise_cache = adapter.create_cache(model)
    multitoken_cache = adapter.create_cache(model)
    forward_kwargs = adapter.kvzip_forward_kwargs()

    with torch.inference_mode():
        # Build identical context state in two independent caches.
        model.model(input_ids=context_ids, past_key_values=tokenwise_cache, use_cache=True)
        model.model(input_ids=context_ids, past_key_values=multitoken_cache, use_cache=True)

        # Transformers 5.2.0's native one-token path is the reference cached
        # continuation behavior for its Qwen3.5 DeltaNet implementation.
        tokenwise_outputs = None
        for token_offset in range(continuation_ids.shape[1]):
            token_position = torch.tensor(
                [[context_tokens + token_offset]],
                dtype=torch.long,
                device=model.device,
            )
            tokenwise_outputs = model(
                input_ids=continuation_ids[:, token_offset : token_offset + 1],
                past_key_values=tokenwise_cache,
                position_ids=token_position,
                use_cache=True,
                **forward_kwargs,
            )

        # This is the KVPress call shape we fixed: the whole continuation is
        # forwarded at once, starting from an already-prefilled hybrid cache.
        multitoken_positions = torch.arange(
            context_tokens,
            context_tokens + continuation_ids.shape[1],
            device=model.device,
        ).unsqueeze(0)
        with adapter.cached_continuation(model):
            multitoken_outputs = model(
                input_ids=continuation_ids,
                past_key_values=multitoken_cache,
                position_ids=multitoken_positions,
                use_cache=True,
                **forward_kwargs,
            )

    tokenwise_logits = tokenwise_outputs.logits[:, -1, :]
    multitoken_logits = multitoken_outputs.logits[:, -1, :]
    tokenwise_token = tokenwise_logits.argmax(dim=-1)
    multitoken_token = multitoken_logits.argmax(dim=-1)
    logits_error = relative_error(tokenwise_logits, multitoken_logits)

    text_config = adapter.get_text_config(model)
    linear_layers = [
        layer_idx for layer_idx, layer_type in enumerate(text_config.layer_types) if layer_type == "linear_attention"
    ]
    recurrent_errors = []
    conv_errors = []
    for layer_idx in linear_layers:
        tokenwise_recurrent = tokenwise_cache.recurrent_states[layer_idx]
        multitoken_recurrent = multitoken_cache.recurrent_states[layer_idx]
        tokenwise_conv = tokenwise_cache.conv_states[layer_idx]
        multitoken_conv = multitoken_cache.conv_states[layer_idx]
        if any(state is None for state in (tokenwise_recurrent, multitoken_recurrent, tokenwise_conv, multitoken_conv)):
            raise AssertionError(f"Missing DeltaNet state at linear-attention layer {layer_idx}")
        recurrent_errors.append(relative_error(tokenwise_recurrent, multitoken_recurrent))
        conv_errors.append(relative_error(tokenwise_conv, multitoken_conv))

    max_recurrent_error = max(recurrent_errors)
    max_conv_error = max(conv_errors)
    print(
        "DeltaNet continuation comparison: "
        f"tokens={continuation_ids.shape[1]}, "
        f"tokenwise_next={tokenwise_token.tolist()}, "
        f"multitoken_next={multitoken_token.tolist()}, "
        f"logits_relative_error={logits_error:.6f}, "
        f"max_recurrent_relative_error={max_recurrent_error:.6f}, "
        f"max_conv_relative_error={max_conv_error:.6f}",
        flush=True,
    )

    # BF16 token-wise and chunk kernels need not be bit-identical. These are
    # deliberately smoke-test tolerances: they catch a missing initial state
    # while permitting ordinary kernel-ordering roundoff.
    if not torch.equal(tokenwise_token, multitoken_token):
        raise AssertionError("Multi-token continuation changed the next-token prediction")
    if max_recurrent_error > 0.10 or max_conv_error > 0.10:
        raise AssertionError(
            "Multi-token continuation diverged from native token-wise DeltaNet state "
            f"(recurrent={max_recurrent_error:.6f}, conv={max_conv_error:.6f})"
        )

    del tokenwise_cache, multitoken_cache, tokenwise_outputs, multitoken_outputs
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/rethinkingai-self/25m0820/kvpress/Qwen3.5-9B",
    )
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--compression-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a GPU compute node.")

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kvpress import KVPressTextGenerationPipeline, KVzipPress
    from kvpress.model_adapter import get_model_adapter

    print(f"Loading model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.name_or_path = args.model

    seed_text = (
        "This is a short Qwen3.5 KVzip smoke-test context. "
        "The important fact is that the adapter must compress only full-attention layers. "
    )
    seed_ids = tokenizer(seed_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    repeats = (args.context_tokens // max(seed_ids.shape[1], 1)) + 1
    context_ids = seed_ids.repeat(1, repeats)[:, : args.context_tokens].to(model.device)

    adapter = get_model_adapter(model)
    cache = adapter.create_cache(model)
    full_layers = [idx for idx, _ in adapter.iter_kv_attention_layers(model)]
    print(f"Full-attention layers: {full_layers}", flush=True)

    print("Verifying DeltaNet cached multi-token continuation...", flush=True)
    verify_delta_multitoken_continuation(model, tokenizer, adapter, seed_ids)

    press = KVzipPress(compression_ratio=args.compression_ratio)
    print("Running KVzip prefill and reconstruction...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    # model.eval() does not disable autograd. Keep the initial prefill and
    # KVzip's context-manager exit/reconstruction inside a no-grad context.
    with torch.no_grad():
        if torch.is_grad_enabled():
            raise AssertionError("Autograd must be disabled during the KVzip smoke test")
        with press(model):
            outputs = model.model(
                input_ids=context_ids,
                past_key_values=cache,
                use_cache=True,
            )
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak allocated GPU memory: {peak_memory_gib:.2f} GiB", flush=True)
    if outputs.past_key_values is not None:
        cache = outputs.past_key_values

    lengths = {
        layer_idx: cache.key_cache[layer_idx].shape[-2]
        for layer_idx in full_layers
        if cache.key_cache[layer_idx] is not None
    }
    print(f"Restored full-attention cache lengths: {lengths}", flush=True)
    if not lengths or not all(length == args.context_tokens for length in lengths.values()):
        raise AssertionError("KVzip left synthetic reconstruction tokens in the full-attention cache")

    masked_counts = {
        layer_idx: len(attention.masked_key_indices[0])
        for layer_idx, attention in adapter.iter_kv_attention_layers(model)
        if getattr(attention, "masked_key_indices", None) is not None
    }
    print(f"KVzip masked K/V positions per full-attention layer: {masked_counts}", flush=True)
    if not masked_counts or sum(masked_counts.values()) == 0:
        raise AssertionError("KVzip did not select any full-attention K/V positions for masking")

    snapshot = adapter.snapshot_cache_state(cache)
    question_ids = tokenizer(
        "What is the important fact?",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    with torch.inference_mode():
        with adapter.cached_continuation(model):
            first = model(input_ids=question_ids, past_key_values=cache, use_cache=True).logits[:, -1, :]
        first_token = first.argmax(dim=-1)
        adapter.restore_cache_state(cache, snapshot)
        with adapter.cached_continuation(model):
            second = model(input_ids=question_ids, past_key_values=cache, use_cache=True).logits[:, -1, :]
        second_token = second.argmax(dim=-1)

    if not torch.equal(first_token, second_token):
        raise AssertionError("Restored cache produced a different first answer token")

    print(f"First generated token after restore: {second_token.tolist()}", flush=True)

    print("Verifying the Qwen3.5 evaluation pipeline...", flush=True)
    pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, device=args.device)

    # Exercise the same pipeline path used by evaluation, including the chat
    # template, KVzip prefill, and greedy answer decoding.
    pipeline_press = KVzipPress(compression_ratio=args.compression_ratio)
    with torch.inference_mode():
        pipeline_answer = pipeline(
            seed_text,
            question="What important fact does the context state?",
            answer_prefix="Final Answer: ",
            press=pipeline_press,
            max_new_tokens=64,
            enable_thinking=False,
        )["answer"]
    print(f"Pipeline answer: {pipeline_answer!r}", flush=True)
    if "<think>" in pipeline_answer or "</think>" in pipeline_answer:
        raise AssertionError("Instruct-mode smoke test produced a thinking block")

    print("Qwen3.5 + KVzip smoke test passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
