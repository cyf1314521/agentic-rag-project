"""
Corpus Gate：query 与会话论文 profile 文本的混合关联度（cosine + BM25Lite，不查 Milvus）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from config import Config

from .query_rewrite import pairs_from_messages
from .session_corpus import SessionCorpus, attach_gate_vectors, load_session_corpus

_TOKEN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass
class ContentGateResult:
    score: float
    best_paper_id: str | None
    method: Literal["hybrid", "embedding", "bm25", "disabled"]
    per_paper_scores: dict[str, float]
    gate_query: str
    embedding_score: float = 0.0
    bm25_score: float = 0.0


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1]


def bm25_lite_score(query: str, document: str) -> float:
    """内存 BM25 近似：查询词在文档中的命中比例（0~1）。"""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    d_tokens = set(_tokenize(document))
    if not d_tokens:
        return 0.0
    hits = sum(1 for t in q_tokens if t in d_tokens)
    return hits / len(q_tokens)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _last_substantive_human(
    recent_messages: list[tuple[str, str]],
) -> str | None:
    from .paper_scope import is_paper_id_only_query
    from .scope import is_meta_not_rag

    for role, content in reversed(recent_messages):
        if role != "human":
            continue
        text = (content or "").strip()
        if not text or is_meta_not_rag(text):
            continue
        if is_paper_id_only_query(text, []):
            continue
        return text
    return None


def build_gate_query(
    raw_query: str,
    *,
    messages: list | None = None,
    pending_user_query: str = "",
    summary: str = "",
    use_prev_turn: bool | None = None,
) -> str:
    """
    构造用于 Corpus Gate 打分的 query（可与 Rewrite 的 effective_query 不同）。
    追问时拼接上一轮实质用户问题。
    """
    q = (raw_query or "").strip()
    if not q:
        return q

    use_prev = (
        Config.CONTENT_GATE_USE_PREV_TURN
        if use_prev_turn is None
        else use_prev_turn
    )
    if not use_prev:
        return q

    pairs = pairs_from_messages(messages or [])
    prev = _last_substantive_human(pairs)
    if not prev and pending_user_query.strip():
        prev = pending_user_query.strip()

    if not prev or prev in q or q in prev:
        return q

    from .query_rewrite import is_anaphoric_followup

    if is_anaphoric_followup(q) or len(q) <= 28:
        return f"{prev} {q}".strip()

    if summary and len(q) <= 40:
        return f"{prev} {q}".strip()

    return q


def score_content_relevance(
    gate_query: str,
    corpus: SessionCorpus,
    embeddings: Any | None,
    *,
    mode: str | None = None,
) -> ContentGateResult:
    """混合打分：默认 0.75 cosine + 0.25 BM25Lite。"""
    mode = (mode or Config.CONTENT_GATE_MODE).lower()
    per_paper: dict[str, float] = {}
    best_pid: str | None = None
    best_score = 0.0
    best_emb = 0.0
    best_bm25 = 0.0

    if not gate_query.strip() or not corpus.reprs:
        return ContentGateResult(
            score=0.0,
            best_paper_id=None,
            method="disabled",
            per_paper_scores={},
            gate_query=gate_query,
        )

    q_vec: list[float] | None = None
    if mode in ("hybrid", "embedding") and embeddings is not None:
        attach_gate_vectors(corpus, embeddings)
        try:
            q_vec = list(embeddings.embed_query(gate_query))
        except Exception:
            q_vec = None

    emb_weight = Config.CONTENT_GATE_EMB_WEIGHT
    bm25_weight = Config.CONTENT_GATE_BM25_WEIGHT

    for repr_ in corpus.reprs:
        emb_s = 0.0
        if q_vec and repr_.gate_vector:
            emb_s = max(0.0, _cosine(q_vec, repr_.gate_vector))
        bm25_s = bm25_lite_score(gate_query, repr_.gate_text)

        if mode == "embedding":
            combined = emb_s
        elif mode == "bm25":
            combined = bm25_s
        else:
            combined = emb_weight * emb_s + bm25_weight * bm25_s

        per_paper[repr_.paper_id] = combined
        if combined > best_score:
            best_score = combined
            best_pid = repr_.paper_id
            best_emb = emb_s
            best_bm25 = bm25_s

    method: Literal["hybrid", "embedding", "bm25", "disabled"] = (
        "hybrid" if mode == "hybrid" else (
            "embedding" if mode == "embedding" else (
                "bm25" if mode == "bm25" else "disabled"
            )
        )
    )

    return ContentGateResult(
        score=best_score,
        best_paper_id=best_pid,
        method=method,
        per_paper_scores=per_paper,
        gate_query=gate_query,
        embedding_score=best_emb,
        bm25_score=best_bm25,
    )


def load_corpus_for_session(paper_ids: list[str]) -> SessionCorpus:
    return load_session_corpus(paper_ids)
