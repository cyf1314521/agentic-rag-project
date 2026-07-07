"""
检索证据统一熔断：rerank / dense 分数门槛，无证据不进入 generate。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document

from config import Config

logger = logging.getLogger(__name__)

EvidenceSignal = Literal["rerank", "dense", "bm25", "none"]


@dataclass
class EvidenceVerdict:
    passed: bool
    top_score: float
    signal: EvidenceSignal
    reason: str


def _doc_score(doc: Document) -> tuple[float, EvidenceSignal]:
    meta = doc.metadata or {}
    if meta.get("rerank_score") is not None:
        try:
            return float(meta["rerank_score"]), "rerank"
        except (TypeError, ValueError):
            pass
    if meta.get("score") is not None:
        try:
            return float(meta["score"]), "dense"
        except (TypeError, ValueError):
            pass
    if meta.get("sparse_score") is not None:
        try:
            return float(meta["sparse_score"]), "bm25"
        except (TypeError, ValueError):
            pass
    return 0.0, "none"


def evaluate_evidence(
    docs: list[Document],
    query: str = "",
    *,
    min_rerank: float | None = None,
    min_fallback: float | None = None,
) -> EvidenceVerdict:
    """判定检索结果是否达到证据门槛。"""
    min_r = min_rerank if min_rerank is not None else Config.RETRIEVAL_MIN_RERANK_SCORE
    min_f = min_fallback if min_fallback is not None else Config.RETRIEVAL_MIN_FALLBACK_SCORE

    if not docs:
        return EvidenceVerdict(
            passed=False,
            top_score=0.0,
            signal="none",
            reason="empty_docs",
        )

    best_score = 0.0
    best_signal: EvidenceSignal = "none"
    has_rerank = False

    for doc in docs:
        score, signal = _doc_score(doc)
        if signal == "rerank":
            has_rerank = True
        if score > best_score:
            best_score = score
            best_signal = signal

    if has_rerank or best_signal == "rerank":
        passed = best_score >= min_r
        reason = "rerank_ok" if passed else "rerank_below_threshold"
        signal = "rerank"
    elif best_signal in ("dense", "bm25"):
        passed = best_score >= min_f
        reason = "fallback_ok" if passed else "fallback_below_threshold"
        signal = best_signal
    else:
        passed = False
        reason = "no_score_metadata"
        signal = "none"
        logger.warning(
            "Evidence gate: %d docs but no rerank/dense scores in metadata",
            len(docs),
        )

    return EvidenceVerdict(
        passed=passed,
        top_score=best_score,
        signal=signal,
        reason=reason,
    )
