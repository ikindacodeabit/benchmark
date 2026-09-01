# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small text statistics shared by the runners and the CSV post-processors.

One definition of "does this prediction contain a thinking block", because
three copies had drifted: the runner, the CSV scanner and the baseline
comparison each re-derived it with their own regexes and column names.
"""

from __future__ import annotations

import re

THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)


def think_tag_stats(text: str) -> dict[str, int | bool]:
    """Count Qwen-style thinking tags in one prediction.

    ``unclosed_think`` means the model started a thinking block the decode
    never finished, which usually indicates the answer was cut off by
    max_new_tokens rather than genuinely produced.
    """
    value = text or ""
    open_tags = len(THINK_OPEN_RE.findall(value))
    close_tags = len(THINK_CLOSE_RE.findall(value))
    return {
        "think_open_tags": open_tags,
        "think_close_tags": close_tags,
        "has_think_tag": bool(open_tags or close_tags),
        "unclosed_think": open_tags > close_tags,
    }


def has_think_tag(text: str) -> bool:
    """Whether a prediction carries any thinking tag, open or close."""
    value = text or ""
    return bool(THINK_OPEN_RE.search(value) or THINK_CLOSE_RE.search(value))
