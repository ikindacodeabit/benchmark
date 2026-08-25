# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process KV-compression backend for RLM sub-calls, via kvpress's own KVzipPress.

`KVzipSubClient` is duck-typed to the surface `rlm.py` consumes from
`NIMClient` (`.chat()`, `.usage`, `.extra_body`, `.model`) and adds
`.chat_split(question, context)`, which routes through
`kvpress.presses.kvzip_press.KVzipPress` and
`kvpress.pipeline.KVPressTextGenerationPipeline` -- the exact same press and
pipeline used for every LOFT/RULER/synthetic-kv benchmark this session, so a
sub-call is compressed the same way and with the same memory-budget knobs
(``--memory-budget``/``--memory-budget-unit``, matching
``matrix_constants.py``'s LOFT budgets: 256MB/512MB/1GB/2GB/4GB) as the rest
of the benchmarking, rather than a separately-tuned fixed ratio.

Deliberate tradeoff vs. the standalone snu-mllab/KVzip backend this replaces:
kvpress's KVzipPress is LOGICAL compression -- evicted keys are masked via an
attention patch, not freed -- so a slice still needs its FULL uncompressed KV
resident in GPU memory regardless of the budget. Unlike standalone KVzip's
EvictCache, this does NOT reduce actual memory usage or let a sub-call afford
a bigger slice for less GPU memory; the memory_budget here only controls how
many tokens are logically retained (same simulated-budget semantics as the
LOFT/RULER matrix runs), not a real memory saving. Chosen for methodological
consistency with the rest of this session's results, not for compute
efficiency.

Heavy imports (torch / transformers / kvpress) are deliberately deferred into
the functions so that importing this module -- e.g. from the CLI wiring in
run_benchmark.py -- stays legal in a venv where they are absent.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

from .client import Usage

SUB_PRESS_CHOICES = ("kvzip", "no_press")

_GIB = 1024**3


# ---- preflight checks -------------------------------------------------------


def query_gpus() -> list[dict]:
    """Free/total memory per GPU via nvidia-smi, WITHOUT initializing torch.

    Runs before any CUDA context exists so that device selection can still be
    done with CUDA_VISIBLE_DEVICES. Returns [] when nvidia-smi is unavailable
    (no NVIDIA driver -- the caller turns that into a clear error).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, free_mib, total_mib = (p.strip() for p in line.split(","))
        gpus.append({"index": int(idx), "free_gib": int(free_mib) / 1024, "total_gib": int(total_mib) / 1024})
    return gpus


def preflight_select_gpu(min_free_gib: float, device: Optional[str] = None) -> int:
    """Verify a GPU with enough free memory exists and pin the process to it.

    Must run BEFORE torch initializes CUDA: pinning works by setting
    CUDA_VISIBLE_DEVICES (the sub model loads with device_map="auto", which
    would otherwise shard it across every visible GPU, including ones other
    jobs are using). Honors an explicit --sub-device / pre-set
    CUDA_VISIBLE_DEVICES rather than second-guessing it, but still checks the
    chosen GPU has room and fails with the per-GPU picture instead of an OOM
    twenty minutes into a run.
    """
    gpus = query_gpus()
    if not gpus:
        raise RuntimeError(
            "No NVIDIA GPU visible (nvidia-smi failed). --sub-backend kvzip loads the sub model "
            "in-process and needs a local GPU; use --sub-backend nim on CPU-only hosts."
        )
    table = "; ".join(f"GPU{g['index']}: {g['free_gib']:.1f}/{g['total_gib']:.1f} GiB free" for g in gpus)

    pinned: Optional[int] = None
    if device and device.startswith("cuda:"):
        pinned = int(device.split(":", 1)[1])
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        # Already pinned externally; check the first visible physical GPU.
        pinned = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

    if pinned is not None:
        match = next((g for g in gpus if g["index"] == pinned), None)
        if match is None:
            raise RuntimeError(f"Requested GPU {pinned} does not exist. Available: {table}")
        if match["free_gib"] < min_free_gib:
            raise RuntimeError(
                f"GPU {pinned} has {match['free_gib']:.1f} GiB free but the sub model needs "
                f"~{min_free_gib:.0f} GiB (weights + full-context KV headroom -- kvpress's "
                f"KVzipPress never frees memory, so this must cover the UNCOMPRESSED context; "
                f"tune with --sub-min-free-gib). Free it, or pick another: {table}"
            )
        chosen = pinned
    else:
        best = max(gpus, key=lambda g: g["free_gib"])
        if best["free_gib"] < min_free_gib:
            raise RuntimeError(
                f"No GPU has the ~{min_free_gib:.0f} GiB free the sub model needs "
                f"(tune with --sub-min-free-gib). {table}"
            )
        chosen = best["index"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    print(f"[kvzip] pinned to GPU {chosen} ({table})")
    return chosen


class KVzipSubClient:
    """Sub-LLM client that generates through kvpress's KVzipPress-compressed cache.

    Loads the model ONCE at construction. Each call builds a fresh
    KVPressTextGenerationPipeline forward pass (prefill+score -> mask -> decode)
    -- kvpress's KVzipPress never frees the masked-out KV, so steady-state GPU
    memory reflects the FULL uncompressed context, not the compressed one.
    """

    def __init__(
        self,
        model: str,
        press_name: str = "kvzip",
        memory_budget: float = 1.0,
        memory_budget_unit: str = "GB",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        max_context_tokens: int = 34000,
        press_min_tokens: int = 1024,
        min_free_gib: float = 14.0,
    ):
        if press_name not in SUB_PRESS_CHOICES:
            raise ValueError(f"unknown press {press_name!r}; choose from {SUB_PRESS_CHOICES}")
        # Order matters: GPU pinning must precede the first torch/CUDA touch,
        # and the check must precede the multi-minute model load.
        preflight_select_gpu(min_free_gib, device)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from kvpress.model_adapter import get_model_adapter
        from kvpress.pipeline import KVPressTextGenerationPipeline
        from kvpress.presses.kvzip_press import KVzipPress

        self.model = model
        self.extra_body: Optional[dict] = None  # assignable no-op (chat_template_kwargs don't apply here)
        self.usage = Usage()
        hf_model = AutoModelForCausalLM.from_pretrained(model, dtype="auto", device_map="auto", trust_remote_code=True)
        hf_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.pipeline = KVPressTextGenerationPipeline(model=hf_model, tokenizer=tokenizer)
        self.press_name = press_name
        self.press = KVzipPress(compression_ratio=0.0) if press_name == "kvzip" else None
        self.memory_budget = memory_budget
        self.memory_budget_unit = memory_budget_unit
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.press_min_tokens = press_min_tokens
        self._example_stats: list[dict] = []
        # Same helper the rest of kvpress uses to convert a GB budget into a
        # per-token byte cost -- reused here instead of re-derived, so this
        # backend's fit check agrees with LOFT/RULER's own budget math.
        self._kv_bytes_per_token = get_model_adapter(hf_model).kv_bytes_per_token(hf_model)
        self._torch = torch

    # ---- NIMClient-compatible surface ---------------------------------------
    def chat(self, messages: list[dict], **kw: Any) -> str:
        """Legacy one-string path (root emitted a plain llm_query(prompt) call).

        The flattened text is treated as KVzip `context` with an empty
        question, so a big single-arg paste is still compressed instead of the
        arm silently degenerating to dense. Tiny prompts (< press_min_tokens)
        skip the press: compressing a two-line question is pure noise.
        """
        text = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("content"))
        return self._generate(context=text, question="", split=False, **kw)

    def chat_split(self, question: str, context: str, system: Optional[str] = None, **kw: Any) -> str:
        """Context-aware path: `context` is compressed, question side is not."""
        q = f"{system}\n\n{question}" if system else question
        return self._generate(context=context, question=q, split=True, **kw)

    # ---- internals ----------------------------------------------------------
    def _fits_in_memory(self, ctx_tokens: int) -> bool:
        """Pre-check that the FULL uncompressed prefill KV fits on the GPU.

        Unlike a real-eviction backend, kvpress's KVzipPress never frees the
        masked-out KV -- the full context's cache is resident for the whole
        call, not just before pruning. 1.2x + 1 GiB covers scoring
        activations and allocator fragmentation; erring cautious here trades a
        retry notice for an OOM that would torch-poison the loaded weights.
        """
        free_bytes, _ = self._torch.cuda.mem_get_info()
        return ctx_tokens * self._kv_bytes_per_token * 1.2 + 1 * _GIB <= free_bytes

    def _generate(self, context: str, question: str, split: bool, **kw: Any) -> str:
        ctx_tokens = self._token_len(context)
        if ctx_tokens > self.max_context_tokens:
            # Token-level analogue of kvpress's max_context_length truncation:
            # a 131072-char slice of dense text can exceed the model window.
            ids = self.pipeline.tokenizer.encode(context, add_special_tokens=False)
            context = self.pipeline.tokenizer.decode(ids[: self.max_context_tokens])
            ctx_tokens = self.max_context_tokens
        prune = self.press is not None and ctx_tokens >= self.press_min_tokens
        if not self._fits_in_memory(ctx_tokens):
            self._torch.cuda.empty_cache()
            if not self._fits_in_memory(ctx_tokens):
                # Returned as the sub-answer (not raised) so the example survives:
                # the root can retry with a smaller slice, exactly like the
                # truncation notice path.
                return "[SUB-MODEL ERROR] the provided text did not fit in GPU memory; retry with a smaller snippet."
        try:
            max_new = int(kw.get("max_tokens") or self.max_new_tokens)
            # Same call shape as every LOFT/RULER benchmark run: memory_budget
            # is converted to a compression_ratio internally by the pipeline,
            # scaled to THIS call's actual context length.
            result = self.pipeline(
                context,
                question=question,
                press=self.press if prune else None,
                memory_budget=self.memory_budget if prune else None,
                memory_budget_unit=self.memory_budget_unit,
                max_new_tokens=max_new,
            )
            answer = str(result["answer"])
            stats = getattr(self.pipeline, "last_memory_budget_stats", {}) if prune else {}
        except self._torch.cuda.OutOfMemoryError:
            return "[SUB-MODEL ERROR] the provided text did not fit in GPU memory; retry with a smaller snippet."
        finally:
            self._torch.cuda.empty_cache()

        self.usage.prompt_tokens += ctx_tokens + self._token_len(question)
        self.usage.completion_tokens += self._token_len(answer)
        self.usage.calls += 1
        self._example_stats.append(
            {
                "split": split,
                "pressed": prune,
                "context_tokens": ctx_tokens,
                "retained_context_tokens": stats.get("retained_context_tokens", ctx_tokens),
                "compression_ratio": stats.get("compression_ratio", 0.0) if prune else 0.0,
            }
        )
        return answer

    def _token_len(self, text: str) -> int:
        if not text:
            return 0
        return len(self.pipeline.tokenizer.encode(text, add_special_tokens=False))

    def pop_example_stats(self) -> dict:
        """Summary of sub-calls since the last pop; the caller attaches it to the
        per-example metrics (as `sub_kv`) and write_run_artifacts aggregates it
        into the run-level analogue of KVPress's average_retained_context_tokens."""
        calls = self._example_stats
        self._example_stats = []
        if not calls:
            return {}
        n = len(calls)
        return {
            "calls": n,
            "split_calls": sum(1 for c in calls if c["split"]),
            "pressed_calls": sum(1 for c in calls if c["pressed"]),
            "average_context_tokens": sum(c["context_tokens"] for c in calls) / n,
            "max_context_tokens": max(c["context_tokens"] for c in calls),
            "average_retained_context_tokens": sum(c["retained_context_tokens"] for c in calls) / n,
            "average_compression_ratio": sum(c["compression_ratio"] for c in calls) / n,
        }
