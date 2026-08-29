# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible client with retries, for a locally served model.

Points at any OpenAI-compatible endpoint — in practice a vLLM (or SGLang)
server on the same host or cluster node. Local servers don't authenticate,
so no API key is involved.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import APIError, APITimeoutError, OpenAI, RateLimitError


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    # The last call's prompt size, as the SERVER counted it. The running totals
    # above cannot answer "how big was the context at its peak", and the RLM's
    # cost axis is exactly that question -- previously answered with a local
    # estimate from a different tokenizer.
    last_prompt_tokens: int | None = None

    def add(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if u:
            self.prompt_tokens += u.prompt_tokens or 0
            self.completion_tokens += u.completion_tokens or 0
            self.last_prompt_tokens = u.prompt_tokens or None
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMClient:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    max_retries: int = 6
    timeout: float = 300.0
    temperature: float = 0.0
    max_tokens: int = 4096
    extra_body: dict | None = None
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        # Local OpenAI-compatible servers ignore the key but the SDK requires one.
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key or "EMPTY", timeout=self.timeout)

    def chat(self, messages: list[dict], **kw) -> str:
        """One chat completion with backoff. Returns assistant text."""
        params = dict(
            model=self.model,
            messages=messages,
            temperature=kw.get("temperature", self.temperature),
            max_tokens=kw.get("max_tokens", self.max_tokens),
        )
        # Per-call extra_body wins; otherwise use the client default (e.g. disable thinking):
        eb = kw.get("extra_body") or self.extra_body
        if eb:
            params["extra_body"] = eb

        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**params)
                self.usage.add(resp)
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120)
            except APIError as e:
                # 5xx are retryable; 4xx (bad request, context too long) are not.
                # Connection errors (e.g. APIConnectionError) have no status_code
                # at all -- treat those as retryable too.
                status = getattr(e, "status_code", None)
                if status is not None and status < 500:
                    raise
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError("unreachable")
