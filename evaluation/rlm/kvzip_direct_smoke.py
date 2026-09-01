# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct, deterministic smoke test for KVzipSubClient's compression path.

Bypasses the full RLM harness entirely (no root model, no vLLM, no model
deciding what to search for) -- constructs a controlled, long-enough context
and a deliberately tiny memory budget, then asserts the KVzip masking code
actually ran and evicted something. This is the thing three RLM-driven smoke
runs each failed to confirm for three different, unrelated reasons (slice too
small, budget too generous, model too erratic) -- none of which say anything
about whether KVzipSubClient itself works.

Not named test_* and not a pytest module: it needs a real GPU and a real model,
so collecting it as a test would fail everywhere the rest of the suite runs.
Run it by hand on a GPU node:

    KVZIP_SMOKE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \\
        python -m evaluation.rlm.kvzip_direct_smoke
"""
import os
import random

from evaluation.rlm.kvzip_backend import KVzipSubClient

# A Hugging Face id or a local path. Defaults to the hub id rather than the
# absolute path of one particular machine's checkout, which nobody else had.
MODEL = os.environ.get("KVZIP_SMOKE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")


def build_context(n_chars: int, needle: str) -> str:
    rng = random.Random(0)
    words = "ocean mountain forest river cloud stone meadow valley harbor lantern".split()
    out, total = [], 0
    while total < n_chars:
        sent = " ".join(rng.choices(words, k=10)).capitalize() + "."
        out.append(sent)
        total += len(sent) + 1
    body = " ".join(out)
    mid = len(body) // 2
    return body[:mid] + f" {needle} " + body[mid:]


def main() -> None:
    needle_value = "8829471"
    context = build_context(n_chars=20000, needle=f"The secret code is {needle_value}.")
    question = "What is the secret code mentioned in the text? Reply with the number only."

    print(f"context length: {len(context)} chars")

    client = KVzipSubClient(
        model=MODEL,
        press_name="kvzip",
        memory_budget=2,
        memory_budget_unit="MB",
        max_new_tokens=32,
        press_min_tokens=0,
        min_free_gib=8.0,
    )
    ctx_tokens = client._token_len(context)
    print(f"context tokens: {ctx_tokens}")

    answer = client.chat_split(question=question, context=context)
    print(f"answer: {answer!r}")

    stats = client.pop_example_stats()
    print(f"stats: {stats}")

    assert stats.get("pressed_calls", 0) == 1, "expected the call to go through the press, it didn't"
    assert stats.get("average_compression_ratio", 0.0) > 0.0, "expected nonzero compression, got 0 -- masking never ran"
    print("PASS: KVzip masking ran and evicted a nonzero fraction of the context.")
    if needle_value in answer:
        print("BONUS: the needle survived compression and the model found it.")
    else:
        print(
            "NOTE: the needle did not appear in the answer -- compression ran, but this specific "
            "budget may have evicted the needle. That's a separate accuracy question, not a wiring bug."
        )


if __name__ == "__main__":
    main()
