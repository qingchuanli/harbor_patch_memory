"""Hybrid BM25 + path-similarity retriever for patch memory records.

Ported verbatim from ``EvoClaw/harness/e2e/patch_memory_retriever.py`` so
ranking semantics match the EvoClaw reference implementation.  Only the
``TYPE_CHECKING`` import line is redirected to the local package.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import FeaturePatchRecord

_SPLIT_RE = re.compile(r"[/_.\-\s]+")


def tokenize(text: str) -> list[str]:
    """Split *text* on path / whitespace delimiters, lower-case, drop short tokens."""
    return [t for t in _SPLIT_RE.split(text.lower()) if len(t) >= 2]


def path_segments(paths: list[str]) -> set[str]:
    """Return individual directory segments **and** cumulative prefixes.

    ``sklearn/ensemble/tests/test_iforest.py`` →
    ``{sklearn, ensemble, tests, test_iforest.py,
       sklearn/ensemble, sklearn/ensemble/tests}``
    """
    segments: set[str] = set()
    for p in paths:
        parts = p.replace("\\", "/").split("/")
        for part in parts:
            if len(part) >= 2:
                segments.add(part)
        for i in range(2, len(parts)):
            segments.add("/".join(parts[:i]))
    return segments


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class _BM25:
    """Tiny BM25-Okapi implementation operating on pre-tokenised documents."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n_docs = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n_docs, 1)

        self.df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return max(math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0), 0.1)

    def score(self, query: list[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        dl = len(doc)
        tf: dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1

        s = 0.0
        for term in query:
            if term not in tf:
                continue
            freq = tf[term]
            idf = self._idf(term)
            numer = freq * (self.k1 + 1.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1))
            s += idf * numer / denom
        return s

    def scores(self, query: list[str]) -> list[float]:
        return [self.score(query, i) for i in range(self.n_docs)]


_STATUS_MULT_EDIT = {
    "validated": 1.3,
    "working": 1.2,
    "submitted": 1.0,
    "failing": 0.6,
    "regressive": 0.4,
    "error": 0.3,
}

_STATUS_MULT_RECOVER = {
    "validated": 1.3,
    "working": 1.4,
    "submitted": 1.0,
    "failing": 0.4,
    "regressive": 0.4,
    "error": 0.3,
}


def _status_multiplier(status: str, mode: str) -> float:
    table = _STATUS_MULT_RECOVER if mode == "recover" else _STATUS_MULT_EDIT
    return table.get(status, 1.0)


def _stage_bonus(source_stage: str) -> float:
    return 1.1 if source_stage == "local_validation" else 1.0


class PatchMemoryRetriever:
    """Rank patch-memory records using BM25 + path-structural similarity."""

    def __init__(self, records: list["FeaturePatchRecord"]):
        self._records = list(records)
        self._bm25: _BM25 | None = None
        self._record_segments: list[set[str]] = []
        self._corpus: list[list[str]] = []
        self._indexed = False

    def _ensure_index(self) -> None:
        if self._indexed:
            return
        self._corpus = [self._tokenize_record(r) for r in self._records]
        self._bm25 = _BM25(self._corpus)
        self._record_segments = [path_segments(r.files_changed) for r in self._records]
        self._indexed = True

    @staticmethod
    def _flatten_to_str(value: object) -> str:
        """Best-effort flatten any value into a plain string for BM25 tokenisation.

        Defensive against LLM-summariser drift (e.g. a list snuck into a
        field that should have been a string).
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(
                PatchMemoryRetriever._flatten_to_str(v) for v in value if v is not None
            )
        return str(value)

    @staticmethod
    def _tokenize_record(record: "FeaturePatchRecord") -> list[str]:
        flat = PatchMemoryRetriever._flatten_to_str
        parts: list[str] = []
        parts.append(flat(record.feature_title))
        for t in record.feature_tags:
            parts.append(flat(t))
        for s in record.symbols_changed:
            parts.append(flat(s))
        for p in record.files_changed:
            parts.append(flat(p))
        parts.append(flat(record.behavior.get("what_changed", "")))
        parts.append(flat(record.behavior.get("why_changed", "")))
        patch = record.code_change.get("patch_text", "")
        if patch:
            parts.append(flat(patch)[:500])
        return tokenize(" ".join(parts))

    def retrieve(
        self,
        active_files: list[str],
        active_features: list[str],
        active_symbols: list[str],
        mode: str = "edit_intent",
        top_k: int = 6,
        min_score: float = 0.15,
        relative_threshold: float = 0.3,
    ) -> list[tuple["FeaturePatchRecord", float]]:
        """Return up to *top_k* records sorted by descending hybrid score."""
        if not self._records:
            return []
        self._ensure_index()
        assert self._bm25 is not None

        query_tokens = tokenize(" ".join(active_files + active_features + active_symbols))
        query_segs = path_segments(active_files)
        active_set = set(active_files)

        bm25_raw = self._bm25.scores(query_tokens)
        max_bm25 = max(bm25_raw) if bm25_raw else 1.0
        if max_bm25 == 0:
            max_bm25 = 1.0

        scored: list[tuple[int, float]] = []
        for i, record in enumerate(self._records):
            psim = jaccard(self._record_segments[i], query_segs)
            exact = 1.0 if set(record.files_changed) & active_set else 0.0
            bm25_norm = bm25_raw[i] / max_bm25

            raw = 0.40 * psim + 0.35 * bm25_norm + 0.25 * exact
            final = (
                raw
                * _status_multiplier(record.status, mode)
                * _stage_bonus(record.source_stage)
            )
            scored.append((i, final))

        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[tuple["FeaturePatchRecord", float]] = []
        for idx, score in scored[:top_k]:
            if score < min_score:
                break
            results.append((self._records[idx], score))

        if results and relative_threshold > 0:
            top_score = results[0][1]
            cutoff = top_score * relative_threshold
            results = [(r, s) for r, s in results if s >= cutoff]

        return results
