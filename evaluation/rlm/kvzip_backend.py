# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process KV-compression backend for RLM sub-calls, via standalone KVzip.

`KVzipSubClient` is duck-typed to the surface `rlm.py` consumes from
`NIMClient` (`.chat()`, `.usage`, `.extra_body`, `.model`) and adds
`.chat_split(question, context)`, which routes through KVzip
(https://github.com/snu-mllab/KVzip): the context is prefilled into an
`EvictCache`, scored by context reconstruction, pruned, and the question is
decoded against the pruned cache.

Why standalone KVzip and not the kvpress library's KVzipPress:
- kvpress's KVzip is LOGICAL compression -- evicted keys are masked via an
  attention patch, not freed -- so a 128k-token slice still needs the FULL
  uncompressed KV in GPU memory. KVzip's EvictCache physically drops the
  evicted pairs before decoding, so compression is a real memory saving.
- KVzip pins transformers==4.51.3, the SAME version the vLLM 0.8.5 root
  server needs, so the driver and a local root server can share one venv
  (kvpress needed >=4.56, forcing the old two-venv split).

Deviations from the NIM path, documented in the loft128k README:
- KVzip renders the context under its own fixed chat template (system prompt
  "You are a helpful assistant." + a QA instruction line); SUB_SYSTEM_PROMPT
  is therefore prepended to the (uncompressed) question side instead of being
  a system message.
- KVzip's `prune(ratio=r)` RETAINS fraction r. The CLI's --compression-ratio
  keeps the kvpress convention (fraction EVICTED), so this module passes
  `ratio = 1 - compression_ratio` and reports stats in the evicted convention.

KVzip is not pip-installable: clone the repo and point the KVZIP_DIR env var
(or --kvzip-dir) at the checkout. It hard-requires flash-attn (its EvictCache
decodes through flash_attn_varlen_func); there is no sdpa fallback.

Heavy imports (torch / transformers / kvzip) are deliberately deferred into
the functions so that importing this module -- e.g. from the CLI wiring in
run_benchmark.py -- stays legal in a venv where they are absent.
"""

from __future__ import annotations

import os
import subprocess
import sys
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
    CUDA_VISIBLE_DEVICES (KVzip loads with device_map="auto", which would
    otherwise shard the sub model across every visible GPU, including ones
    other jobs are using). Honors an explicit --sub-device / pre-set
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
                f"~{min_free_gib:.0f} GiB (weights + KV headroom; tune with --sub-min-free-gib). "
                f"Free it, or pick another: {table}"
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


def import_kvzip(kvzip_dir: Optional[str] = None) -> Any:
    """Import ModelKVzip from a KVzip checkout, with actionable failure modes.

    KVzip has no pip package; its top-level package is the generic name
    `model`, so the checkout dir goes on sys.path. flash-attn is checked first
    because its absence otherwise surfaces as an ImportError deep inside
    KVzip's attention module.
    """
    path = kvzip_dir or os.environ.get("KVZIP_DIR")
    if path:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(os.path.join(path, "model", "wrapper.py")):
            raise RuntimeError(f"--kvzip-dir/KVZIP_DIR={path} is not a KVzip checkout (no model/wrapper.py)")
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import flash_attn  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "KVzip requires flash-attn (its evict cache decodes through flash_attn_varlen_func; "
            "there is no sdpa fallback). Install with: "
            "pip install flash-attn==2.7.4.post1 --no-build-isolation"
        ) from e
    try:
        import tiny_api_cuda  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "KVzip's CUDA kernel (tiny_api_cuda) is not installed; its evict cache needs it "
            "to flatten pruned KV. Build it with: cd <KVzip checkout>/csrc && python build.py install "
            "(do NOT `pip install -e .` the KVzip repo itself — it force-downgrades torch to 2.3.0)."
        ) from e
    try:
        from model import ModelKVzip
    except ImportError as e:
        raise RuntimeError(
            "Could not import KVzip. Clone https://github.com/snu-mllab/KVzip, "
            "`pip install -r requirements.txt` in it, and point --kvzip-dir (or the "
            "KVZIP_DIR env var) at the checkout."
        ) from e
    return ModelKVzip


class KVzipSubClient:
    """Sub-LLM client that generates through a KVzip-pruned KV cache.

    Loads the model ONCE at construction (~8 GB bf16 for Qwen3-4B). Each call
    builds a fresh EvictCache (prefill -> reconstruction scoring -> prune ->
    decode) and frees it afterwards, so steady-state GPU memory returns to the
    weights between calls.
    """

    def __init__(
        self,
        model: str,
        press_name: str = "kvzip",
        compression_ratio: float = 0.5,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        max_context_tokens: int = 34000,
        press_min_tokens: int = 1024,
        kvzip_dir: Optional[str] = None,
        min_free_gib: float = 14.0,
    ):
        if press_name not in SUB_PRESS_CHOICES:
            raise ValueError(f"unknown press {press_name!r}; choose from {SUB_PRESS_CHOICES}")
        # Order matters: GPU pinning must precede the first torch/CUDA touch,
        # and both checks must precede the multi-minute model download/load.
        preflight_select_gpu(min_free_gib, device)
        ModelKVzip = import_kvzip(kvzip_dir)

        self.model = model
        self.extra_body: Optional[dict] = None  # assignable no-op (chat_template_kwargs don't apply here)
        self.usage = Usage()
        self.kvzip = ModelKVzip(model)  # kv_type="evict": pruned pairs are physically freed
        self.press_name = press_name
        self.compression_ratio = compression_ratio
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.press_min_tokens = press_min_tokens
        self._example_stats: list[dict] = []
        cfg = self.kvzip.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        # K + V, bf16: bytes of cache per context token, for the per-call fit check.
        self._kv_bytes_per_token = 2 * 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim

    # ---- NIMClient-compatible surface ---------------------------------------
    def chat(self, messages: list[dict], **kw: Any) -> str:
        """Legacy one-string path (root emitted a plain llm_query(prompt) call).

        The flattened text is treated as KVzip `context` with an empty
        question, so a big single-arg paste is still compressed instead of the
        arm silently degenerating to dense. Tiny prompts (< press_min_tokens)
        skip the prune: compressing a two-line question is pure noise.
        """
        text = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("content"))
        return self._generate(context=text, question="", split=False, **kw)

    def chat_split(self, question: str, context: str, system: Optional[str] = None, **kw: Any) -> str:
        """Context-aware path: `context` is compressed, question side is not."""
        q = f"{system}\n\n{question}" if system else question
        return self._generate(context=context, question=q, split=True, **kw)

    # ---- internals ----------------------------------------------------------
    def _fits_in_memory(self, ctx_tokens: int) -> bool:
        """Pre-check that the uncompressed prefill KV fits on the GPU.

        Pruning only frees memory AFTER prefill+scoring, so the fit check is
        against the FULL cache. 1.2x + 1 GiB covers scoring activations and
        allocator fragmentation; erring cautious here trades a retry notice for
        an OOM that would torch-poison the loaded weights.
        """
        import torch

        free_bytes, _ = torch.cuda.mem_get_info()
        return ctx_tokens * self._kv_bytes_per_token * 1.2 + 1 * _GIB <= free_bytes

    def _generate(self, context: str, question: str, split: bool, **kw: Any) -> str:
        import torch

        ctx_tokens = self._token_len(context)
        if ctx_tokens > self.max_context_tokens:
            # Token-level analogue of kvpress's max_context_length truncation:
            # a 131072-char slice of dense text can exceed the model window.
            ids = self.kvzip.tokenizer.encode(context, add_special_tokens=False)
            context = self.kvzip.tokenizer.decode(ids[: self.max_context_tokens])
            ctx_tokens = self.max_context_tokens
        prune = self.press_name != "no_press" and ctx_tokens >= self.press_min_tokens
        if not self._fits_in_memory(ctx_tokens):
            torch.cuda.empty_cache()
            if not self._fits_in_memory(ctx_tokens):
                # Returned as the sub-answer (not raised) so the example survives:
                # the root can retry with a smaller slice, exactly like the
                # truncation notice path.
                return "[SUB-MODEL ERROR] the provided text did not fit in GPU memory; retry with a smaller snippet."
        kv = None
        try:
            # do_score is the reconstruction-scoring pass (~2-3x prefill cost);
            # pointless work when nothing will be pruned.
            kv = self.kvzip.prefill(context, load_score=False, do_score=prune)
            retained_frac = 1.0
            if prune:
                _, retained_frac = kv.prune(ratio=1.0 - self.compression_ratio)
            query_ids = self.kvzip.apply_template(question)
            max_new = int(kw.get("max_tokens") or self.max_new_tokens)
            self.kvzip.gen_kwargs["max_new_tokens"] = max_new
            answer = str(self.kvzip.generate(query_ids, kv=kv, update_cache=False))
        except torch.cuda.OutOfMemoryError:
            return "[SUB-MODEL ERROR] the provided text did not fit in GPU memory; retry with a smaller snippet."
        finally:
            del kv
            torch.cuda.empty_cache()

        self.usage.prompt_tokens += ctx_tokens + self._token_len(question)
        self.usage.completion_tokens += self._token_len(answer)
        self.usage.calls += 1
        self._example_stats.append(
            {
                "split": split,
                "pressed": prune,
                "context_tokens": ctx_tokens,
                "retained_context_tokens": int(round(retained_frac * ctx_tokens)),
                # Stat keeps the kvpress convention: fraction of context KV evicted.
                "compression_ratio": (1.0 - retained_frac) if prune else 0.0,
            }
        )
        return answer

    def _token_len(self, text: str) -> int:
        if not text:
            return 0
        return len(self.kvzip.tokenizer.encode(text, add_special_tokens=False))

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
