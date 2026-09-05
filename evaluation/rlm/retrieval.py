# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A BM25 index over fixed windows of the RLM's document.

The root model's only way to locate anything was ``str.find``, which returns the
FIRST occurrence. In a 3.7M-character LOFT corpus a term like "cart" or "1983"
occurs hundreds of times, so the first hit is close to a random probe -- and
nothing sweeps the document when it misses. Measured over the 879 finished
LOFT-1m transcripts, 80.5% of examples never surfaced the gold answer into any
REPL output at all, against 5.2% that saw it and still answered wrong.

Deliberately free of torch/transformers imports, for the same reason
``sizing.py`` is: ``run_benchmark.py`` must stay importable in a venv without
them, since the ``http`` sub-call path has no torch.

TWO PROPERTIES THE REST OF THE HARNESS DEPENDS ON:

1. **Hits are verbatim slices of the document, carried as offsets.** ``Hit.text``
   is exactly ``document[hit.start:hit.end]``. That is what lets a retrieved
   payload be attributed to real document spans by ``record_coverage``, and what
   keeps ``_expand_subcall``'s "locate the slice verbatim" invariant satisfiable.
   Returning copies of text, or normalized text, would silently break both.
2. **The index holds offsets and postings, never chunk text.** The document is
   kept by reference. A 1M-token corpus chunked at 2000 characters is ~2300
   windows; materializing those as strings would duplicate the whole corpus.

Scope, measured rather than assumed: this is a large win where the corpus is a
collection of independent passages and the question carries lexical anchors
(LOFT nq/hotpotqa, longbench 2wikimqa/hotpotqa: 83-95% recall@10). It is a much
smaller win on a single coherent document with discourse-level questions
(longbench qasper: 41% recall@5), and it does nothing for aggregation tasks
(RULER cwe, variable tracking) where the right strategy is to read everything.
So this is offered to the root as ONE MORE TOOL, never as a replacement for
sweeping or for peeking at structure.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

DEFAULT_CHUNK_CHARS = 2000
DEFAULT_OVERLAP = 400
DEFAULT_K = 5

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Dropped before scoring. Deliberately short and closed-class: a longer list
# starts removing terms that carry the query (e.g. "who" in a WHO-the-agency
# question), and BM25's idf already discounts anything this common.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those it its of in on at to for from by with as
    is are was were be been being am do does did doing have has had having
    what when where which who whom whose why how
    """.split()
)


@dataclass(frozen=True)
class Hit:
    """One retrieved window, addressed by its offsets into the document."""

    rank: int
    start: int
    end: int
    score: float
    text: str

    def __repr__(self) -> str:  # pragma: no cover - display only
        preview = self.text[:60].replace("\n", " ")
        return f"Hit(rank={self.rank}, start={self.start}, end={self.end}, score={self.score:.2f}, {preview!r}...)"


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, minus stopwords and single characters."""
    return [w for w in _TOKEN_RE.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


class ChunkIndex:
    """BM25 over fixed-width, overlapping character windows of one document.

    Overlap matters: a gold span straddling a window boundary would otherwise be
    split across two chunks and match neither well.
    """

    def __init__(
        self,
        document: str,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        overlap: int = DEFAULT_OVERLAP,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if chunk_chars <= 0:
            raise ValueError(f"chunk_chars must be positive, got {chunk_chars}")
        if not 0 <= overlap < chunk_chars:
            raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_chars, got {overlap} with {chunk_chars}")

        self.document = document
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.k1 = k1
        self.b = b

        stride = chunk_chars - overlap
        self.spans: list[tuple[int, int]] = [
            (start, min(start + chunk_chars, len(document))) for start in range(0, max(len(document), 1), stride)
        ]

        self._tf: list[Counter] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        document_frequency: Counter = Counter()

        for index, (start, end) in enumerate(self.spans):
            counts = Counter(tokenize(document[start:end]))
            self._tf.append(counts)
            self._lengths.append(sum(counts.values()))
            for term in counts:
                document_frequency[term] += 1
                self._postings[term].append(index)

        self.n_chunks = len(self.spans)
        self._avg_length = (sum(self._lengths) / self.n_chunks) if self.n_chunks else 0.0
        self._idf: dict[str, float] = {
            term: math.log(1 + (self.n_chunks - freq + 0.5) / (freq + 0.5)) for term, freq in document_frequency.items()
        }

    def search(self, query: str, k: int = DEFAULT_K) -> list[Hit]:
        """The k best-scoring windows for `query`, best first."""
        if k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index in self._postings[term]:
                freq = self._tf[index][term]
                length = self._lengths[index]
                denominator = freq + self.k1 * (1 - self.b + self.b * length / (self._avg_length or 1.0))
                scores[index] += idf * freq * (self.k1 + 1) / denominator

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
        hits = []
        for rank, (index, score) in enumerate(ranked, start=1):
            start, end = self.spans[index]
            hits.append(Hit(rank=rank, start=start, end=end, score=score, text=self.document[start:end]))
        return hits


# Size-1 cache, module level so the harness and the offline recall script share it.
#
# The LOFT corpus is byte-identical across every example of a subset -- all 110
# nq_1m rows carry the same 3,723,430 characters -- so a per-example build would
# pay 0.35s per example for the same index. Exactly ONE entry, so switching
# documents drops the previous index instead of accumulating multi-megabyte
# indexes per subset; peak is one document plus its index.
_CACHED_INDEX: Optional[ChunkIndex] = None


def get_index(
    document: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> ChunkIndex:
    """The index for this document, reusing the previous one when it matches.

    Compare document contents: loaders can produce equal corpora as distinct
    string objects. The index already holds its document and window settings.
    """
    global _CACHED_INDEX
    if (
        _CACHED_INDEX is not None
        and _CACHED_INDEX.chunk_chars == chunk_chars
        and _CACHED_INDEX.overlap == overlap
        and _CACHED_INDEX.document == document
    ):
        return _CACHED_INDEX
    _CACHED_INDEX = ChunkIndex(document, chunk_chars=chunk_chars, overlap=overlap)
    return _CACHED_INDEX


def reset_cache() -> None:
    """Drop the cached index. For tests, and to free a large index explicitly."""
    global _CACHED_INDEX
    _CACHED_INDEX = None
