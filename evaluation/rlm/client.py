# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible client for NVIDIA NIM with rate limiting + retries.

Works with the hosted catalog (https://integrate.api.nvidia.com/v1) or any
self-hosted NIM/vLLM endpoint — just change base_url in the config.
"""
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field

from openai import OpenAI, APIError, RateLimitError, APITimeoutError


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if u:
            self.prompt_tokens += u.prompt_tokens or 0
            self.completion_tokens += u.completion_tokens or 0
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RateLimiter:
    """Simple thread-safe limiter: max N requests per 60s window."""

    def __init__(self, rpm: int = 35):  # stay under NIM's ~40 rpm free tier
        self.min_interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = self._last + self.min_interval - now
            if delta > 0:
                time.sleep(delta)
            self._last = time.monotonic()


@dataclass
class NIMClient:
    model: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: str | None = None
    rpm: int = 35
    max_retries: int = 6
    timeout: float = 300.0
    temperature: float = 0.0
    max_tokens: int = 4096
    extra_body: dict | None = None
    # Optional shared limiter: pass ONE RateLimiter to several clients (e.g. the
    # RLM root + sub) so the rpm cap applies per ACCOUNT, not per client — two
    # independent limiters would otherwise let the process issue ~2x rpm.
    limiter: RateLimiter | None = None
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("NVIDIA_API_KEY")
        # Only the hosted NVIDIA catalog actually authenticates; a local
        # vLLM/NIM server accepts any string, so don't make a laptop/infolab run
        # fail on a key it never needed.
        if not key:
            if "api.nvidia.com" in self.base_url:
                raise RuntimeError(
                    "NVIDIA_API_KEY is not set, and base_url points at the hosted NVIDIA catalog "
                    f"({self.base_url}), which requires one (get it at build.nvidia.com). "
                    "If you meant a local server, pass --base-url http://localhost:8000/v1 — "
                    "no key is needed there."
                )
            key = "EMPTY"  # placeholder; local OpenAI-compatible servers ignore it
        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=self.timeout)
        self._limiter = self.limiter or RateLimiter(self.rpm)

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
            self._limiter.wait()
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
                # at all -- treat those as retryable too. Exception: NIM returns
                # 400 "DEGRADED function cannot be invoked" while an endpoint is
                # unhealthy -- that's a transient service condition, retry it.
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and "DEGRADED" not in str(e):
                    raise
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError("unreachable")
