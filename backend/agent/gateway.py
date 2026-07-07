"""
企业级前置网关：Meta 硬拦 + 任务型离域 + Corpus Gate（session-relative）。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from config import Config

from .content_gate import (
    ContentGateResult,
    build_gate_query,
    load_corpus_for_session,
    score_content_relevance,
)
from .scope import is_meta_not_rag, looks_like_paper_rag
from .session_corpus import SessionCorpus

GatewayAction = Literal["continue", "out_of_scope", "upload_required"]

_TASK_OFF_DOMAIN = re.compile(
    r"写个|帮我写|生成|实现|debug|代码|python|javascript|脚本|程序|"
    r"translate|翻译|写诗|笑话|段子|天气|温度|下雨|预报|"
    r"股价|股票|大盘|汇率|"
    r"help me code|write (a |an )?.+ script|implement .+ in|stock price|weather",
    re.I,
)


@dataclass
class GatewayDecision:
    action: GatewayAction
    reason: str
    gate_query: str = ""
    content_score: float | None = None
    best_paper_id: str | None = None
    content_result: ContentGateResult | None = None
    latency_ms: float = 0.0
    trace: dict = field(default_factory=dict)


def is_task_off_domain(query: str) -> bool:
    """任务型离域（非话题黑名单）：写代码、天气、股价等。"""
    return bool(_TASK_OFF_DOMAIN.search((query or "").strip()))


def _legacy_rule_gate(query: str, session_paper_ids: list[str]) -> GatewayAction:
    """CONTENT_GATE_ENABLED=false 时回退旧规则网关。"""
    from .scope import _legacy_scope_gate_action

    act = _legacy_scope_gate_action(query, session_paper_ids)
    return "continue" if act == "continue" else "out_of_scope"


def run_gateway(
    raw_query: str,
    session_paper_ids: list[str],
    *,
    messages: list | None = None,
    pending_user_query: str = "",
    summary: str = "",
    embeddings: Any | None = None,
    corpus: SessionCorpus | None = None,
) -> GatewayDecision:
    """
    同步前置网关（Rewrite 之前调用）。

    顺序：empty → meta → looks_like fast-path → task+低分 → corpus gate
    """
    t0 = time.perf_counter()
    q = (raw_query or "").strip()

    if not q:
        return GatewayDecision(
            action="out_of_scope",
            reason="empty_query",
            gate_query=q,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    if not session_paper_ids:
        return GatewayDecision(
            action="upload_required",
            reason="no_session",
            gate_query=q,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    if not Config.CONTENT_GATE_ENABLED:
        action = _legacy_rule_gate(q, session_paper_ids)
        return GatewayDecision(
            action=action,
            reason="legacy_gate",
            gate_query=q,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    if is_meta_not_rag(q):
        return GatewayDecision(
            action="out_of_scope",
            reason="meta",
            gate_query=q,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    gate_query = build_gate_query(
        q,
        messages=messages,
        pending_user_query=pending_user_query,
        summary=summary,
    )

    if looks_like_paper_rag(q, session_paper_ids):
        return GatewayDecision(
            action="continue",
            reason="looks_like_fastpath",
            gate_query=gate_query,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    session_corpus = corpus or load_corpus_for_session(session_paper_ids)
    content = score_content_relevance(
        gate_query,
        session_corpus,
        embeddings,
    )

    min_score = Config.CONTENT_GATE_MIN_SCORE
    high_score = Config.CONTENT_GATE_HIGH_SCORE

    if Config.TASK_OFF_DOMAIN_ENABLED and is_task_off_domain(q):
        override_min = Config.TASK_OVERRIDE_MIN_SCORE
        if content.score >= override_min:
            return GatewayDecision(
                action="continue",
                reason="task_override",
                gate_query=gate_query,
                content_score=content.score,
                best_paper_id=content.best_paper_id,
                content_result=content,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        return GatewayDecision(
            action="out_of_scope",
            reason="task_off_domain",
            gate_query=gate_query,
            content_score=content.score,
            best_paper_id=content.best_paper_id,
            content_result=content,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    if content.score >= high_score:
        action: GatewayAction = "continue"
        reason = "content_high"
    elif content.score >= min_score:
        action = "continue"
        reason = "content_ok"
    else:
        action = "out_of_scope"
        reason = "low_relevance"

    return GatewayDecision(
        action=action,
        reason=reason,
        gate_query=gate_query,
        content_score=content.score,
        best_paper_id=content.best_paper_id,
        content_result=content,
        latency_ms=(time.perf_counter() - t0) * 1000,
        trace={
            "method": content.method,
            "embedding_score": content.embedding_score,
            "bm25_score": content.bm25_score,
            "per_paper_scores": content.per_paper_scores,
        },
    )
