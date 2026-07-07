"""向后兼容：请使用 agent.scope、agent.gateway 与 agent.turn_resolver。"""

from .scope import (
    looks_like_paper_rag,
    scope_gate,
    scope_check_pre_llm,
    is_meta_not_rag,
    upload_required_reply,
    out_of_scope_reply,
    scope_hint_message,
)
from .gateway import run_gateway, GatewayDecision, is_task_off_domain

enterprise_fallback_reply = out_of_scope_reply


def is_standalone_off_topic(query: str, session_paper_ids: list[str]) -> bool:
    decision = run_gateway(query, session_paper_ids)
    return decision.action != "continue"


def resolve_out_of_scope(
    query: str,
    session_paper_ids: list[str],
    intent,
    *,
    threshold: float | None = None,
):
    from .intent import RetrievalIntent
    from .turn_resolver import TurnState, resolve_turn

    if not isinstance(intent, RetrievalIntent):
        intent = RetrievalIntent(
            effective_query=query,
            intent="need_clarification",
            missing=[],
            confidence=float(getattr(intent, "confidence", 0.0) or 0.0),
        )
    decision = run_gateway(query, session_paper_ids)
    r = resolve_turn(
        intent,
        query,
        session_paper_ids,
        raw_query=query,
        scope_continue=(decision.action == "continue"),
    )
    if r.get("turn_state") in (TurnState.UPLOAD_REQUIRED.value, TurnState.OUT_OF_SCOPE.value):
        return {
            "intent": "out_of_domain",
            "answer": r["answer"],
            "match_reason": r["match_reason"],
            "direct_response": True,
            "needs_clarification": False,
            "retrieval_mode": "none",
        }
    return None
