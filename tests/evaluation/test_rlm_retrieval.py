# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BM25 index the RLM root searches with, and the span arithmetic around it.

Pure: no client, no torch, no GPU -- in the style of test_rlm_sizing.py.
"""

import inspect

import pytest

from evaluation.rlm import retrieval
from evaluation.rlm.rlm import _clip_spans, _merge_spans, _widen_spans

FILLER = "the quick brown fox jumps over the lazy dog and keeps on running through fields "
NEEDLE = "Bignotti Cotter Racing entered the Datsun sponsored round at Riverside. "


def make_document(needle_at: int = 40) -> tuple[str, int]:
    """Filler with one distinctive sentence planted at a known offset."""
    head = FILLER * needle_at
    return head + NEEDLE + FILLER * 60, len(head)


@pytest.fixture(autouse=True)
def _clean_cache():
    retrieval.reset_cache()
    yield
    retrieval.reset_cache()


class TestIndexShape:
    def test_windows_tile_the_document_at_the_configured_stride(self):
        document = "x" * 10_000
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        starts = [start for start, _ in index.spans]
        assert starts[:3] == [0, 1600, 3200]
        assert all(end <= len(document) for _, end in index.spans)
        assert index.spans[-1][1] == len(document)

    def test_a_short_document_is_one_window(self):
        index = retrieval.ChunkIndex("only a little text here", chunk_chars=2000, overlap=400)
        assert index.spans == [(0, len("only a little text here"))]

    def test_an_empty_document_does_not_explode(self):
        index = retrieval.ChunkIndex("", chunk_chars=2000, overlap=400)
        assert index.search("anything", 5) == []

    def test_overlap_must_be_smaller_than_the_window(self):
        with pytest.raises(ValueError):
            retrieval.ChunkIndex("abc", chunk_chars=100, overlap=100)


class TestSearchQuality:
    def test_the_top_hit_contains_the_planted_rare_term(self):
        document, offset = make_document()
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        hits = index.search("Bignotti Cotter Datsun Riverside", 5)
        assert hits, "a distinctive query returned nothing"
        assert hits[0].start <= offset < hits[0].end

    def test_hit_text_is_a_verbatim_slice_of_the_document(self):
        """Everything downstream rests on this: coverage attribution, the
        min-subcall floor, and `document.find` all need hits to be real slices."""
        document, _ = make_document()
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        for hit in index.search("Datsun Riverside round", 5):
            assert hit.text == document[hit.start : hit.end]

    def test_a_term_straddling_a_boundary_is_still_found(self):
        """The whole reason `overlap` exists."""
        document = ("a " * 990) + "ZEBRAQUUX PECULIAR " + ("b " * 990)
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        hits = index.search("ZEBRAQUUX PECULIAR", 3)
        assert hits and "ZEBRAQUUX" in hits[0].text

    def test_a_term_in_every_window_does_not_outrank_a_rare_one(self):
        document, offset = make_document()
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        # "fox" is in every window; "Bignotti" is in exactly one.
        hits = index.search("fox Bignotti", 5)
        assert hits[0].start <= offset < hits[0].end

    def test_a_query_of_only_unknown_terms_returns_nothing(self):
        document, _ = make_document()
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        assert index.search("xyzzy plugh frobnicate", 5) == []

    def test_ranks_are_dense_and_ordered(self):
        document, _ = make_document()
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        hits = index.search("quick brown fox", 4)
        assert [h.rank for h in hits] == [1, 2, 3, 4]
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_the_same_query_twice_gives_identical_spans(self):
        """A grid cell has to be reproducible, so ties must break stably."""
        document, _ = make_document()
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        first = [(h.start, h.end) for h in index.search("quick brown fox", 6)]
        assert first == [(h.start, h.end) for h in index.search("quick brown fox", 6)]

    def test_k_bounds_the_number_of_hits(self):
        document, _ = make_document()
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        assert len(index.search("quick brown fox", 3)) == 3
        assert index.search("quick brown fox", 0) == []

    def test_the_repr_is_short_enough_to_print(self):
        """`for h in hits: print(h)` is the idiom the prompt teaches, and the
        observation is truncated at obs_limit (6000 chars by default)."""
        document, _ = make_document()
        index = retrieval.ChunkIndex(document, chunk_chars=2000, overlap=400)
        for hit in index.search("Bignotti Datsun", 5):
            assert len(repr(hit)) < 200


class TestIndexCache:
    def test_two_equal_but_distinct_documents_build_one_index(self):
        """The LOFT case, and the reason `id()` was rejected as a cache key: the
        loaders rebuild each row's context from Arrow, so all 110 rows of a subset
        carry equal-but-distinct 3.7MB strings."""
        document, _ = make_document()
        # Built by concatenation, since CPython returns the SAME object for
        # "".join([s]) and for s + "" -- neither would exercise the cache.
        twin = document[:10] + document[10:]
        assert twin is not document and twin == document
        before = retrieval.build_count()
        retrieval.get_index(document, 2000, 400)
        retrieval.get_index(twin, 2000, 400)
        assert retrieval.build_count() - before == 1

    def test_a_different_document_evicts_the_previous_index(self):
        first, _ = make_document()
        second = first + " and something else entirely to change the content "
        before = retrieval.build_count()
        one = retrieval.get_index(first, 2000, 400)
        two = retrieval.get_index(second, 2000, 400)
        assert two is not one
        assert retrieval.build_count() - before == 2
        # One slot: going back rebuilds rather than serving a retained second entry.
        retrieval.get_index(first, 2000, 400)
        assert retrieval.build_count() - before == 3

    def test_changing_the_geometry_rebuilds(self):
        document, _ = make_document()
        before = retrieval.build_count()
        retrieval.get_index(document, 2000, 400)
        retrieval.get_index(document, 4000, 400)
        assert retrieval.build_count() - before == 2


class TestNoDeepLearningStack:
    def test_the_module_imports_no_torch_or_transformers(self):
        """run_benchmark.py must stay importable in a venv without them; the http
        sub-call path has no torch. Same contract sizing.py states."""
        imports = [
            line.strip()
            for line in inspect.getsource(retrieval).splitlines()
            if line.startswith(("import ", "from "))
        ]
        assert imports, "no imports found -- did the module move?"
        for banned in ("torch", "transformers", "numpy", "scipy", "sklearn"):
            assert not any(banned in line for line in imports), f"{banned} reached retrieval.py"


class TestSpanArithmetic:
    def test_merge_coalesces_overlapping_and_touching_spans(self):
        assert _merge_spans([(0, 100), (80, 200)]) == [(0, 200)]
        assert _merge_spans([(0, 100), (100, 200)]) == [(0, 200)]
        assert _merge_spans([(300, 400), (0, 100)]) == [(0, 100), (300, 400)]

    def test_merge_is_what_stops_overlapping_windows_double_counting(self):
        """Adjacent retrieval windows share `overlap` characters by construction."""
        spans = [(0, 2000), (1600, 3600)]
        assert _merge_spans(spans) == [(0, 3600)]
        assert sum(e - s for s, e in _merge_spans(spans)) == 3600
        assert sum(e - s for s, e in spans) == 4000  # the double count avoided

    def test_clip_keeps_only_what_survived_truncation(self):
        spans = [(0, 100), (500, 600), (900, 1000)]
        assert _clip_spans(spans, 100, 0) == [(0, 100)]
        assert _clip_spans(spans, 150, 0) == [(0, 100), (500, 550)]
        assert _clip_spans(spans, 0, 0) == []
        assert _clip_spans(spans, 10_000, 0) == spans

    def test_clip_charges_for_the_separator_between_spans(self):
        spans = [(0, 100), (500, 600)]
        assert _clip_spans(spans, 105, 5) == [(0, 100)]

    def test_widen_reaches_the_floor_and_stays_in_the_document(self):
        widened = _widen_spans([(1000, 1100)], target_chars=500, doc_len=5000)
        assert sum(e - s for s, e in widened) >= 500
        assert all(0 <= s < e <= 5000 for s, e in widened)

    def test_widen_at_the_document_edge_still_reaches_the_floor(self):
        widened = _widen_spans([(0, 50)], target_chars=400, doc_len=5000)
        assert widened[0][0] == 0
        assert sum(e - s for s, e in widened) >= 400

    def test_widen_leaves_an_already_large_span_alone(self):
        assert _widen_spans([(0, 900)], target_chars=500, doc_len=5000) == [(0, 900)]

    def test_widen_cannot_exceed_a_short_document(self):
        widened = _widen_spans([(0, 10)], target_chars=10_000, doc_len=40)
        assert widened == [(0, 40)]
