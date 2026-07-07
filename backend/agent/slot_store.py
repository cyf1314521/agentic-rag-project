"""
跨轮槽位继承：focus / intent / constraints 写入 checkpoint slot_memory。
"""

from __future__ import annotations

from typing import Any

from config import Config

from .states import SlotMemory


def empty_slot_memory() -> SlotMemory:
    return {
        "focus_paper_ids": [],
        "intent": "",
        "constraints": {},
        "last_effective_query": "",
        "confirmed_at_turn": "",
    }


def should_clear_slot_memory(
    *,
    direct_response: bool,
    rewrite_reason: str = "",
    turn_state: str = "",
) -> bool:
    if direct_response:
        return True
    if rewrite_reason == "topic_switch":
        return True
    if turn_state in ("upload_required", "out_of_scope"):
        return True
    return False


def pre_inherit_intent_focus(
    intent: Any,
    slot_memory: SlotMemory | None,
    session_paper_ids: list[str],
) -> Any:
    """在 SlotValidate 之前把上轮 focus 写回 IntentLLM 结果。"""
    if not Config.SLOT_INHERITANCE_ENABLED or not slot_memory:
        return intent
    if intent.focus_paper_ids:
        return intent
    if "which_paper" in (intent.missing or []):
        return intent

    session = set(session_paper_ids)
    inherited = [
        p for p in (slot_memory.get("focus_paper_ids") or []) if p in session
    ]
    if not inherited and len(session_paper_ids) == 1:
        inherited = [session_paper_ids[0]]
    if not inherited:
        return intent

    return intent.model_copy(update={"focus_paper_ids": inherited})


def inherit_focus_from_memory(
    validated: dict,
    slot_memory: SlotMemory | None,
    session_paper_ids: list[str],
) -> dict:
    """LLM focus 为空时，从 slot_memory 继承（仅当非 which_paper 澄清）。"""
    if not Config.SLOT_INHERITANCE_ENABLED:
        return validated
    if not slot_memory:
        return validated
    if validated.get("needs_clarification"):
        return validated

    missing = list(validated.get("missing_slots") or [])
    if "which_paper" in missing:
        return validated

    focus = list(validated.get("focus_paper_ids") or [])
    if focus:
        return validated

    session = set(session_paper_ids)
    inherited = [
        p for p in (slot_memory.get("focus_paper_ids") or []) if p in session
    ]
    if not inherited and len(session_paper_ids) == 1:
        inherited = [session_paper_ids[0]]

    if not inherited:
        return validated

    out = dict(validated)
    out["focus_paper_ids"] = inherited
    if out.get("turn_state") == "need_clarification" and out.get("intent") == "paper_qa":
        out["turn_state"] = "rag_ready"
        out["needs_clarification"] = False
        out["match_reason"] = "slot_inherited"
    return out


def update_slot_memory(
    validated: dict,
    slot_memory: SlotMemory | None,
    *,
    turn_id: str = "",
    rewrite_reason: str = "",
) -> SlotMemory:
    if not Config.SLOT_INHERITANCE_ENABLED:
        return slot_memory or empty_slot_memory()

    if should_clear_slot_memory(
        direct_response=bool(validated.get("direct_response")),
        rewrite_reason=rewrite_reason,
        turn_state=str(validated.get("turn_state", "")),
    ):
        return empty_slot_memory()

    if validated.get("needs_clarification"):
        return slot_memory or empty_slot_memory()

    if validated.get("turn_state") != "rag_ready":
        return slot_memory or empty_slot_memory()

    focus = list(validated.get("focus_paper_ids") or [])
    return {
        "focus_paper_ids": focus,
        "intent": str(validated.get("intent", "paper_qa")),
        "constraints": dict(validated.get("constraints") or {}),
        "last_effective_query": str(validated.get("query", "")),
        "confirmed_at_turn": turn_id or "",
    }


def apply_slot_lifecycle(
    validated: dict,
    slot_memory: SlotMemory | None,
    session_paper_ids: list[str],
    *,
    turn_id: str = "",
    rewrite_reason: str = "",
) -> tuple[dict, SlotMemory]:
    validated = inherit_focus_from_memory(validated, slot_memory, session_paper_ids)
    new_memory = update_slot_memory(
        validated,
        slot_memory,
        turn_id=turn_id,
        rewrite_reason=rewrite_reason,
    )
    return validated, new_memory
