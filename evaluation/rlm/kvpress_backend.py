# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process KV-compression backend for RLM sub-calls.

`KVPressSubClient` is duck-typed to the surface `rlm.py` consumes from
`NIMClient` (`.chat()`, `.usage`, `.extra_body`, `.model`) and adds
`.chat_split(question, context)`, which routes through kvpress's
`KVPressTextGenerationPipeline` so the press compresses ONLY the context's KV
during prefill while the question stays uncompressed.

Why in-process rather than a server: the vLLM 0.8.5 *server* needs
transformers==4.51.3, while kvpress needs >=4.56 (the new Cache API). The
benchmark driver imports neither vLLM nor transformers on its own, so it runs
in the kvpress venv, talks to the vLLM ROOT server over HTTP, and hosts the
compressed SUB model here directly — which also gives us the per-call
retained-KV stats without inventing a wire format for them.

Two deviations from the NIM path, both documented in the loft128k README:
- The kvpress pipeline renders the context as a single user turn and has no
  system-role support, so SUB_SYSTEM_PROMPT is prepended to the (uncompressed)
  question side instead of being a system message.
- KVzip in this repo is LOGICAL compression (evicted keys are masked via the
  attention patch, not freed), so retained-token stats are quality knobs, not
  memory savings; budget GPU memory for the full uncompressed KV.

Heavy imports (torch / transformers / kvpress) are deliberately deferred into
the functions so that importing this module — e.g. from the CLI wiring in
run_benchmark.py — stays legal in the RLM venv where they are absent or too
old.
"""

from __future__ import annotations

from typing import Any, Optional

from .client import Usage

SUB_PRESS_CHOICES = ("kvzip", "kvzip_plus", "snapkv", "fastkvzip", "no_press")


def build_press(press_name: str, compression_ratio: float) -> Optional[Any]:
    """Construct a press by registry-style name and assign its ratio.

    Mirrors evaluation/evaluate_registry.py: presses are built bare and the
    ratio is assigned afterwards. `no_press` returns None (dense sub-calls
    through the same pipeline — the control for the press itself).
    """
    if press_name == "no_press":
        return None
    from kvpress import FastKVzipPress, KVzipPress, SnapKVPress

    if press_name == "kvzip":
        press: Any = KVzipPress()
    elif press_name == "kvzip_plus":
        press = KVzipPress(kvzip_plus_normalization=True)
    elif press_name == "snapkv":
        press = SnapKVPress()
    elif press_name == "fastkvzip":
        press = FastKVzipPress()
    else:
        raise ValueError(f"unknown press {press_name!r}; choose from {SUB_PRESS_CHOICES}")
    press.compression_ratio = compression_ratio
    return press


class KVPressSubClient:
    """Sub-LLM client that generates through a press-compressed HF model.

    Loads the model ONCE at construction (~8 GB bf16 for Qwen3-4B) and reuses
    one press instance across calls — safe because presses reset their internal
    state per invocation and kvpress's attention patch clears
    `masked_key_indices` on every fresh prefill.
    """

    def __init__(
        self,
        model: str,
        press_name: str = "kvzip",
        compression_ratio: float = 0.5,
        device: Optional[str] = None,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 512,
        max_context_tokens: int = 34000,
        press_min_tokens: int = 1024,
    ):
        import torch
        from transformers import pipeline as hf_pipeline

        import kvpress  # noqa: F401  (registers the pipeline task; patches attention)

        self.model = model
        self.extra_body: Optional[dict] = None  # assignable no-op (chat_template_kwargs don't apply here)
        self.usage = Usage()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # KVzip asserts a non-eager attention implementation; sdpa satisfies it
        # without requiring flash-attn on the infolab hosts.
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.pipe: Any = hf_pipeline(
            "kv-press-text-generation",
            model=model,
            device=self.device,
            model_kwargs={"attn_implementation": attn_implementation, "dtype": dtype},
            trust_remote_code=True,
        )
        self.pipe.model.eval()
        self.press = build_press(press_name, compression_ratio)
        self.press_name = press_name
        self.compression_ratio = compression_ratio
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.press_min_tokens = press_min_tokens
        self._example_stats: list[dict] = []

    # ---- NIMClient-compatible surface ---------------------------------------
    def chat(self, messages: list[dict], **kw: Any) -> str:
        """Legacy one-string path (root emitted a plain llm_query(prompt) call).

        The flattened text is treated as pipeline `context` with an empty
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
    def _generate(self, context: str, question: str, split: bool, **kw: Any) -> str:
        import torch

        press = self.press
        # Both paths: pressing a tiny context is measurement noise, and some
        # presses hard-require a minimum context (SnapKV asserts the query length
        # exceeds its 64-token window) -- the default floor of 1024 keeps every
        # supported press out of its crash zone. The per-call `pressed` stat
        # records the skip.
        if press is not None and self._token_len(context) < self.press_min_tokens:
            press = None
        try:
            out = self.pipe(
                context,
                question=question,
                press=press,
                max_new_tokens=int(kw.get("max_tokens") or self.max_new_tokens),
                max_context_length=self.max_context_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Returned as the sub-answer (not raised) so the example survives:
            # the root can retry with a smaller slice, exactly like the
            # truncation notice path.
            return "[SUB-MODEL ERROR] the provided text did not fit in GPU memory; retry with a smaller snippet."
        answer = str(out["answer"])

        stats = dict(getattr(self.pipe, "last_memory_budget_stats", None) or {})
        ctx_tokens = int(stats.get("context_tokens", 0)) or self._token_len(context)
        self.usage.prompt_tokens += ctx_tokens + self._token_len(question)
        self.usage.completion_tokens += self._token_len(answer)
        self.usage.calls += 1
        self._example_stats.append(
            {
                "split": split,
                "pressed": press is not None,
                "context_tokens": ctx_tokens,
                "retained_context_tokens": int(stats.get("retained_context_tokens", ctx_tokens)),
                "compression_ratio": float(stats.get("compression_ratio", 0.0)) if press is not None else 0.0,
            }
        )
        return answer

    def _token_len(self, text: str) -> int:
        if not text:
            return 0
        return len(self.pipe.tokenizer.encode(text, add_special_tokens=False))

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
