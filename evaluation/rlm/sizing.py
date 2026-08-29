# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive the RLM sub-call chunk size from a KV memory budget.

The chunk size used to be a hand-picked constant (32000 chars, or 131072 for the
"bigger reads" arm) chosen independently of ``--memory-budget`` -- two settings
describing one constraint, free to contradict each other. This module inverts
kvpress's own budget arithmetic so the chunk follows from the budget instead.

kvpress goes budget -> ratio: ``retained = min(ctx_len, token_budget)`` and
``ratio = 1 - retained/ctx_len`` (``KVPressTextGenerationPipeline._compute_context_compression_ratio``).
Reading that backwards, a chunk of ``token_budget / (1 - target_ratio)`` tokens is
the one whose realized ratio lands on ``target_ratio`` -- so the target ratio is
what picks the chunk, and the budget fixes the scale.

TWO THINGS THIS DOES NOT DO, both easy to misread:

1. It does not save memory. kvpress's ``KVzipPress`` is LOGICAL compression --
   evicted keys are masked, never freed -- so a chunk costs its FULL uncompressed
   KV for the whole call regardless of the budget. ``gpu_fit`` is therefore a cap
   on the full chunk, and it is usually the binding one. Sizing here buys a
   constant, comparable realized ratio across budgets, not headroom.
2. It does not guarantee the target ratio is reached. The size is *advertised* to
   the root model in its system prompt; the root is free to send smaller slices,
   and ``rlm.py`` truncates on the way up but never pads. The realized ratio is
   whatever ``metrics.json``'s ``average_sub_compression_ratio`` reports.

Deliberately free of torch/transformers imports: ``run_benchmark.py`` must stay
importable in a venv without them (the ``http`` path has no torch), and the run
directory name depends on these types on every path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# Mirrors of the runtime fit check in kvzip_backend._fits_in_memory. Shared so the
# planner and the per-call check cannot drift apart: the 1.2x covers scoring
# activations and allocator fragmentation, the 1 GiB is flat slack.
KV_FIT_SAFETY_FACTOR = 1.2
KV_FIT_HEADROOM_BYTES = 1 * 1024**3

# Same assumption as TokenCounter's fallback in rlm.py. Only used when calibration
# is impossible -- LOFT prose runs ~3.6-4.2 chars/token, but densely-tokenising
# subsets (RULER cwe, synthetic-kv) run nearer 2.5, and guessing high there makes
# the sub client silently truncate.
DEFAULT_CHARS_PER_TOKEN = 4.0
CHARS_PER_TOKEN_BOUNDS = (1.5, 6.0)

# Headroom inside the sub model's window for the question and the decoded answer:
# kvzip_backend._generate truncates only the CONTEXT to max_context_tokens, then
# appends the question and decodes max_new_tokens on top of it. Nothing else
# budgets for those two.
DEFAULT_RESERVE_TOKENS = 1024

# Below this a "chunk" is not worth a sub-call: the press is skipped anyway
# (press_min_tokens) and the root is better off reading the slice itself.
DEFAULT_MIN_TOKENS = 1024


@dataclass(frozen=True)
class SubcallSizing:
    """A resolved chunk size plus everything needed to explain it after the fact.

    Serialized wholesale into the run's ``config.yaml``. ``binding`` and ``caps``
    are the audit trail: a run that quietly hit the model window rather than its
    budget looks identical in the size alone.
    """

    chars: int
    tokens: int
    binding: str  # "budget" | "sub_window" | "cli_cap" | "gpu_fit"
    target_compression_ratio: float
    token_budget: int
    kv_bytes_per_token: int
    memory_budget: float
    memory_budget_unit: str
    uncompressed_tokens: int  # what the budget alone asked for, before clamping
    chars_per_token: float
    chars_per_token_source: str  # "calibrated" | "clamped" | "fallback"
    realized_ratio_if_filled: float
    caps: dict = field(default_factory=dict)

    def describe(self) -> str:
        """One-line summary for the run log."""
        return (
            f"{self.chars} chars ({self.tokens} tok), binding={self.binding}, "
            f"budget={self.token_budget} tok, target_ratio={self.target_compression_ratio:g}, "
            f"ratio_if_filled={self.realized_ratio_if_filled:.3f}, "
            f"chars/token={self.chars_per_token:.2f} ({self.chars_per_token_source})"
        )


def gpu_fit_token_cap(
    free_bytes: int,
    kv_bytes_per_token: int,
    safety_factor: float = KV_FIT_SAFETY_FACTOR,
    headroom_bytes: int = KV_FIT_HEADROOM_BYTES,
) -> int:
    """Largest chunk whose FULL uncompressed KV fits in ``free_bytes``.

    The exact inverse of kvzip_backend's per-call check
    ``ctx_tokens * kv_bytes_per_token * 1.2 + 1 GiB <= free_bytes``. Full, not
    retained, because KVzipPress masks rather than frees: the whole chunk's cache
    is resident for the duration of the call.

    Note ``free_bytes`` is a device-global reading, so a co-tenant process on the
    same GPU changes it -- which is why the resolved size is recorded per run
    rather than assumed reproducible.
    """
    if kv_bytes_per_token < 1:
        raise ValueError(f"kv_bytes_per_token must be positive, got {kv_bytes_per_token}")
    usable = free_bytes - headroom_bytes
    if usable <= 0:
        return 0
    return int(usable / (kv_bytes_per_token * safety_factor))


def calibrate_chars_per_token(
    sample: str,
    encode: Callable[[str], Sequence[int]],
    *,
    sample_chars: int = 200_000,
    bounds: tuple[float, float] = CHARS_PER_TOKEN_BOUNDS,
    fallback: float = DEFAULT_CHARS_PER_TOKEN,
) -> tuple[float, str]:
    """Measure chars-per-token on a real document instead of assuming 4.

    Samples from the MIDDLE of the text: LOFT and RULER contexts both open with
    structural boilerplate (instructions, corpus headers) that tokenizes nothing
    like the body the root will actually slice.

    Returns ``(value, source)`` where source is "calibrated", "clamped" (the
    measurement was real but implausible, so it was pulled to a bound), or
    "fallback" (no usable measurement).
    """
    low, high = bounds
    if not sample:
        return fallback, "fallback"

    if len(sample) > sample_chars:
        mid = len(sample) // 2
        half = sample_chars // 2
        sample = sample[mid - half : mid + half]

    try:
        n_tokens = len(encode(sample))
    except Exception:
        # A tokenizer failure must not take the run down over a sizing heuristic.
        return fallback, "fallback"
    if n_tokens < 1:
        return fallback, "fallback"

    measured = len(sample) / n_tokens
    if measured < low:
        return low, "clamped"
    if measured > high:
        return high, "clamped"
    return measured, "calibrated"


def size_subcall_chunk(
    *,
    token_budget: int,
    target_compression_ratio: float,
    chars_per_token: float,
    kv_bytes_per_token: int,
    memory_budget: float,
    memory_budget_unit: str,
    chars_per_token_source: str = "calibrated",
    sub_window_tokens: Optional[int] = None,
    cli_max_context_tokens: Optional[int] = None,
    gpu_free_bytes: Optional[int] = None,
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
) -> SubcallSizing:
    """Resolve the advertised sub-call chunk size.

    ``token_budget`` comes from ``kvpress.pipeline.compute_token_budget_from_memory``
    -- reused rather than re-derived, so this agrees with what the press will
    actually do at call time.

    The budget sets the ask; the smallest live cap wins and is recorded as
    ``binding``. Every cap considered is kept in ``caps`` so a surprising size can
    be explained without re-running.
    """
    if not 0.0 <= target_compression_ratio < 1.0:
        raise ValueError(
            f"target_compression_ratio must be in [0.0, 1.0), got {target_compression_ratio}. "
            "A ratio of 1.0 would ask for an infinitely large chunk."
        )
    if token_budget < 1:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if chars_per_token <= 0:
        raise ValueError(f"chars_per_token must be positive, got {chars_per_token}")

    # target 0.0 degenerates to the budget itself: a chunk that fits uncompressed.
    uncompressed_tokens = int(token_budget // (1.0 - target_compression_ratio))

    caps: dict = {
        "budget": uncompressed_tokens,
        # max(0, ...): a reserve larger than the window used to go NEGATIVE here,
        # win the min() below, and get relabeled "floor" -- advertising a chunk
        # the GPU provably cannot serve.
        "sub_window": None if sub_window_tokens is None else max(0, sub_window_tokens - reserve_tokens),
        "cli_cap": None if cli_max_context_tokens is None else max(0, cli_max_context_tokens - reserve_tokens),
        "gpu_fit": (
            None if gpu_free_bytes is None else gpu_fit_token_cap(gpu_free_bytes, kv_bytes_per_token)
        ),
    }
    live = {name: value for name, value in caps.items() if value is not None}
    binding = min(live, key=lambda name: live[name])
    tokens = live[binding]

    if tokens < min_tokens:
        # Every cap is implausibly tight (a nearly-full GPU, a tiny window).
        # Proceeding used to relabel this "floor" and advertise min_tokens
        # anyway -- and every sub-call then came back [SUB-MODEL ERROR]. An
        # unusable configuration should stop the run before it burns hours.
        table = ", ".join(f"{name}={value}" for name, value in caps.items())
        raise RuntimeError(
            f"auto chunk sizing came out at {tokens} tokens (binding {binding!r}), below the "
            f"usable minimum of {min_tokens}. Caps considered: {table}. Free GPU memory, raise "
            "--sub-max-context-tokens, or lower --subcall-reserve-tokens."
        )

    return SubcallSizing(
        chars=int(tokens * chars_per_token),
        tokens=tokens,
        binding=binding,
        target_compression_ratio=target_compression_ratio,
        token_budget=token_budget,
        kv_bytes_per_token=kv_bytes_per_token,
        memory_budget=memory_budget,
        memory_budget_unit=memory_budget_unit,
        uncompressed_tokens=uncompressed_tokens,
        chars_per_token=chars_per_token,
        chars_per_token_source=chars_per_token_source,
        # What the press would report IF the root filled the chunk. Same formula as
        # the pipeline's _compute_context_compression_ratio, so the two agree.
        realized_ratio_if_filled=1.0 - (min(tokens, token_budget) / tokens),
        caps=caps,
    )
