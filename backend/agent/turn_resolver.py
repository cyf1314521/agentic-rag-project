"""
方案 B — 四轮状态机：唯一路由出口。

UPLOAD_REQUIRED → OUT_OF_SCOPE → NEED_CLARIFICATION → RAG_READY
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from config import Config
from .states import SlotMemory
from .slot_store import apply_slot_lifecycle, pre_inherit_intent_focus
from .intent import RetrievalIntent, IntentKind
from .paper_scope import format_clarification_message
from .scope import (
    upload_required_reply,
    out_of_scope_reply,
)
from .text_utils import query_language

_EVAL_NUM = re.compile(r"eval_(\d+)", re.I)

ScopeMode = Literal[
    "no_session",
    "single_session",
    "explicit",
    "session_wide",
    "discovery",
    "ambiguous",
]


class TurnState(str, Enum):
    UPLOAD_REQUIRED = "upload_required"
    OUT_OF_SCOPE = "out_of_scope"
    NEED_CLARIFICATION = "need_clarification"
    RAG_READY = "rag_ready"


def _clarification_kind(missing: list[str]) -> str:
    slots = set(missing)
    if "which_paper" in slots and "what_to_ask" in slots:
        return "both"
    if "what_to_ask" in slots:
        return "intent"
    if "which_paper" in slots:
        return "paper"
    return "paper"


def _default_clarification(
    query: str,
    session_paper_ids: list[str],
    missing: list[str],
) -> str:
    lang = query_language(query)
    papers = "\n".join(f"- `{pid}`" for pid in session_paper_ids)
    slots = set(missing)

    if "which_paper" in slots and "what_to_ask" in slots:
        if lang == "zh":
            return (
                "请同时说明要查询的论文和具体问题。\n\n"
                f"本会话可用论文：\n{papers}"
            )
        return (
            "Please specify which paper and what you want to know.\n\n"
            f"Papers in session:\n{papers}"
        )

    if "what_to_ask" in slots:
        if lang == "zh":
            return (
                "您只提供了论文标识，我还需要知道您想了解的具体方面。\n\n"
                f"本会话可用论文：\n{papers}\n\n"
                "示例：「eval_05 的摘要说了什么？」"
            )
        return (
            "You provided a paper id but not what to look up.\n\n"
            f"Papers in session:\n{papers}"
        )

    if "which_paper" in slots:
        return format_clarification_message(query, session_paper_ids)

    if lang == "zh":
        return "我无法从当前对话确定您的意图，请重新说明要查询的论文和问题。"
    return (
        "I cannot determine your intent from the conversation. "
        "Please clarify which paper and what you want to know."
    )


def _derive_scope_mode(
    intent_kind: IntentKind,
    focus: list[str],
    session_count: int,
) -> ScopeMode:
    if session_count == 0:
        return "no_session"
    if session_count == 1:
        if intent_kind == "paper_discovery":
            return "discovery"
        return "single_session"
    if len(focus) == 1:
        return "explicit"
    if intent_kind == "paper_compare":
        return "session_wide"
    if intent_kind == "paper_discovery":
        return "discovery"
    if intent_kind == "paper_qa" and not focus:
        return "session_wide"
    return "ambiguous"


def _eval_sort_key(paper_id: str) -> int:
    m = _EVAL_NUM.search(paper_id)
    return int(m.group(1)) if m else 0


def _filter_candidates_by_constraints(
    session_paper_ids: list[str],
    constraints: dict[str, str],
    query: str,
) -> list[str]:
    candidates = list(session_paper_ids)
    if not constraints:
        return candidates

    version = (constraints.get("version") or "").strip()
    if version:
        v = version.lower().lstrip("v")
        narrowed = [
            pid for pid in candidates
            if v in pid.lower() or v in (query or "").lower()
        ]
        if narrowed:
            candidates = narrowed

    time_pref = (constraints.get("time") or "").strip().lower()
    if time_pref in ("latest", "newest", "recent", "最新", "最近"):
        candidates = sorted(candidates, key=_eval_sort_key, reverse=True)

    topic = (constraints.get("topic") or "").strip().lower()
    if topic:
        narrowed = [pid for pid in candidates if topic in pid.lower()]
        if narrowed:
            candidates = narrowed

    return candidates


def _slot_validate(
    intent: RetrievalIntent,
    query: str,
    session_paper_ids: list[str],
    *,
    threshold: float,
) -> dict:
    """SlotValidate 阶段：仅在域内检查槽位是否足够检索。"""
    intent_kind: IntentKind = intent.intent
    missing = list(intent.missing or [])
    focus = [p for p in (intent.focus_paper_ids or []) if p in session_paper_ids]
    needs_clarification = False
    match_reason = "ok"
    constraints = dict(intent.constraints or {})
    single_pid = session_paper_ids[0] if len(session_paper_ids) == 1 else ""

    if len(session_paper_ids) == 1:
        if missing or intent_kind == "need_clarification":
            needs_clarification = True
            match_reason = "missing_slots"
        elif intent.confidence < threshold:
            needs_clarification = True
            match_reason = "low_confidence"
        else:
            intent_kind = "paper_qa" if intent_kind == "need_clarification" else intent_kind

        if intent_kind == "paper_qa":
            focus = [single_pid]
        elif intent_kind == "paper_discovery":
            focus = []
        else:
            focus = focus or ([single_pid] if intent_kind == "paper_compare" else [])

        scope_mode: ScopeMode = _derive_scope_mode(
            intent_kind, focus, len(session_paper_ids),
        )
        if needs_clarification:
            scope_mode = "ambiguous"
    else:
        if missing:
            needs_clarification = True
            match_reason = "missing_slots"
        elif intent_kind == "need_clarification":
            needs_clarification = True
            match_reason = "llm_need_clarification"
            if not missing:
                missing = ["which_paper"]
        elif intent.confidence < threshold:
            needs_clarification = True
            match_reason = "low_confidence"
        elif intent_kind == "paper_qa" and not focus:
            needs_clarification = True
            match_reason = "no_focus_multi_paper"
            missing = ["which_paper"]

        scope_mode = _derive_scope_mode(intent_kind, focus, len(session_paper_ids))
        if needs_clarification:
            scope_mode = "ambiguous"

    clarification_message = ""
    if needs_clarification:
        clarification_message = (
            intent.clarification_question.strip()
            or _default_clarification(query, session_paper_ids, missing)
        )

    retrieval_mode = (
        "profile" if intent_kind == "paper_discovery" and not needs_clarification else "body"
    )

    if needs_clarification:
        return {
            "turn_state": TurnState.NEED_CLARIFICATION.value,
            "intent": intent_kind,
            "focus_paper_ids": [],
            "scope_mode": scope_mode,
            "needs_clarification": True,
            "direct_response": False,
            "answer": "",
            "clarification_message": clarification_message,
            "clarification_kind": _clarification_kind(missing),
            "candidate_paper_ids": (
                _filter_candidates_by_constraints(session_paper_ids, constraints, query)
                if "which_paper" in missing
                else []
            ),
            "match_reason": match_reason,
            "retrieval_mode": retrieval_mode,
            "missing_slots": missing,
            "constraints": constraints,
        }

    return {
        "turn_state": TurnState.RAG_READY.value,
        "intent": intent_kind,
        "focus_paper_ids": focus,
        "scope_mode": scope_mode,
        "needs_clarification": False,
        "direct_response": False,
        "answer": "",
        "clarification_message": "",
        "clarification_kind": "",
        "candidate_paper_ids": [],
        "match_reason": match_reason,
        "retrieval_mode": retrieval_mode,
        "missing_slots": missing,
        "constraints": constraints,
    }


def _direct_pack(
    turn_state: TurnState,
    *,
    query: str,
    session_paper_ids: list[str],
    answer: str,
    intent: RetrievalIntent | None = None,
) -> dict:
    return {
        "query": query,
        "turn_state": turn_state.value,
        "intent": "out_of_domain",
        "focus_paper_ids": [],
        "scope_mode": "no_session" if not session_paper_ids else "session_wide",
        "needs_clarification": False,
        "direct_response": True,
        "answer": answer,
        "clarification_message": "",
        "clarification_kind": "",
        "candidate_paper_ids": list(session_paper_ids),
        "match_reason": turn_state.value,
        "retrieval_mode": "none",
        "missing_slots": [],
        "constraints": dict(intent.constraints or {}) if intent else {},
    }


def resolve_turn(
    intent: RetrievalIntent,
    effective_query: str,
    session_paper_ids: list[str],
    *,
    raw_query: str = "",
    scope_continue: bool = True,
    slot_memory: SlotMemory | None = None,
    turn_id: str = "",
    rewrite_reason: str = "",
) -> dict:
    """
    方案 B 唯一路由 API。

    调用前须已完成：CheckSession、Gateway、Rewrite、effective_query、IntentLLM。
    scope_continue=False 表示 Gateway 已判定离域，直接返回 OUT_OF_SCOPE。
    """
    threshold = Config.INTENT_CONFIDENCE_THRESHOLD
    display_query = (raw_query or effective_query).strip() or effective_query
    eq = (effective_query or "").strip()
    ood_query = display_query or eq

    # Phase 1: CheckSession
    if not session_paper_ids:
        result = _direct_pack(
            TurnState.UPLOAD_REQUIRED,
            query=display_query,
            session_paper_ids=[],
            answer=upload_required_reply(display_query),
            intent=intent,
        )
        result, new_mem = apply_slot_lifecycle(
            result, slot_memory, session_paper_ids,
            turn_id=turn_id, rewrite_reason=rewrite_reason,
        )
        result["slot_memory"] = new_mem
        return result

    if not scope_continue:
        result = _direct_pack(
            TurnState.OUT_OF_SCOPE,
            query=display_query,
            session_paper_ids=session_paper_ids,
            answer=out_of_scope_reply(ood_query, session_paper_ids),
            intent=intent,
        )
        result, new_mem = apply_slot_lifecycle(
            result, slot_memory, session_paper_ids,
            turn_id=turn_id, rewrite_reason=rewrite_reason,
        )
        result["slot_memory"] = new_mem
        return result

    # SlotValidate（仅 Gateway 已放行的在域流量）
    intent = pre_inherit_intent_focus(intent, slot_memory, session_paper_ids)
    slot = _slot_validate(intent, eq, session_paper_ids, threshold=threshold)
    result = {
        "query": eq,
        **slot,
    }
    result, new_mem = apply_slot_lifecycle(
        result, slot_memory, session_paper_ids,
        turn_id=turn_id, rewrite_reason=rewrite_reason,
    )
    result["slot_memory"] = new_mem
    return result
