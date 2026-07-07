"""槽位继承与 slot_memory 生命周期测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.intent import RetrievalIntent
from agent.slot_store import (
    apply_slot_lifecycle,
    empty_slot_memory,
    inherit_focus_from_memory,
    should_clear_slot_memory,
    update_slot_memory,
)
from agent.turn_resolver import resolve_turn


SESSION = ["eval_03_blockchain_scm", "eval_05_wind_turbine"]


def test_inherit_focus_when_llm_empty():
    mem = {
        "focus_paper_ids": ["eval_03_blockchain_scm"],
        "intent": "paper_qa",
        "constraints": {},
        "last_effective_query": "TPS",
        "confirmed_at_turn": "t1",
    }
    validated = {
        "focus_paper_ids": [],
        "turn_state": "rag_ready",
        "intent": "paper_qa",
        "needs_clarification": False,
        "missing_slots": [],
    }
    out = inherit_focus_from_memory(validated, mem, SESSION)
    assert out["focus_paper_ids"] == ["eval_03_blockchain_scm"]
    assert out.get("match_reason") == "slot_inherited"


def test_no_inherit_when_which_paper_missing():
    mem = empty_slot_memory()
    mem["focus_paper_ids"] = ["eval_03_blockchain_scm"]
    validated = {
        "focus_paper_ids": [],
        "needs_clarification": True,
        "missing_slots": ["which_paper"],
        "intent": "paper_qa",
    }
    out = inherit_focus_from_memory(validated, mem, SESSION)
    assert out["focus_paper_ids"] == []


def test_clear_on_topic_switch():
    assert should_clear_slot_memory(
        direct_response=False,
        rewrite_reason="topic_switch",
    )


def test_clear_on_direct_response():
    assert should_clear_slot_memory(direct_response=True)


def test_preserve_on_clarification():
    validated = {
        "needs_clarification": True,
        "turn_state": "need_clarification",
        "direct_response": False,
        "focus_paper_ids": [],
        "intent": "paper_qa",
        "constraints": {},
        "query": "哪篇？",
    }
    mem = {
        "focus_paper_ids": ["eval_05_wind_turbine"],
        "intent": "paper_qa",
        "constraints": {},
        "last_effective_query": "TPS",
        "confirmed_at_turn": "t0",
    }
    new_mem = update_slot_memory(validated, mem)
    assert new_mem["focus_paper_ids"] == ["eval_05_wind_turbine"]


def test_update_on_rag_ready():
    validated = {
        "needs_clarification": False,
        "turn_state": "rag_ready",
        "direct_response": False,
        "focus_paper_ids": ["eval_03_blockchain_scm"],
        "intent": "paper_qa",
        "constraints": {"year": "2024"},
        "query": "共识机制",
    }
    _, new_mem = apply_slot_lifecycle(
        validated, empty_slot_memory(), SESSION, turn_id="t2"
    )
    assert new_mem["focus_paper_ids"] == ["eval_03_blockchain_scm"]
    assert new_mem["confirmed_at_turn"] == "t2"


def test_resolve_turn_multi_paper_inherits_focus():
    intent = RetrievalIntent(
        effective_query="共识机制是什么",
        intent="paper_qa",
        focus_paper_ids=[],
        missing=[],
        confidence=0.9,
    )
    mem = {
        "focus_paper_ids": ["eval_03_blockchain_scm"],
        "intent": "paper_qa",
        "constraints": {},
        "last_effective_query": "TPS",
        "confirmed_at_turn": "t1",
    }
    out = resolve_turn(
        intent,
        intent.effective_query,
        SESSION,
        scope_continue=True,
        slot_memory=mem,
        turn_id="t2",
    )
    assert out["focus_paper_ids"] == ["eval_03_blockchain_scm"]
    assert out["turn_state"] == "rag_ready"
    assert out["slot_memory"]["focus_paper_ids"] == ["eval_03_blockchain_scm"]


def test_pre_inherit_single_paper_default():
    from agent.slot_store import pre_inherit_intent_focus

    intent = RetrievalIntent(
        effective_query="P99",
        intent="paper_qa",
        focus_paper_ids=[],
        missing=[],
        confidence=0.9,
    )
    updated = pre_inherit_intent_focus(intent, empty_slot_memory(), ["eval_05_wind_turbine"])
    assert updated.focus_paper_ids == ["eval_05_wind_turbine"]
