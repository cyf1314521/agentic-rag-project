"""Unit tests for thin intent validation (slot + confidence, no topic guessing)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.intent import RetrievalIntent
from agent.intent_validate import validate_intent

SESSION = [
    "eval_03_blockchain_scm",
    "eval_05_wind_turbine",
    "eval_10_quantum_error",
]


def test_single_paper_session_forces_focus_for_qa():
    intent = RetrievalIntent(
        effective_query="摘要说了什么？",
        intent="paper_qa",
        focus_paper_ids=[],
        missing=[],
        confidence=0.9,
    )
    out = validate_intent(intent, intent.effective_query, ["eval_05_wind_turbine"])
    assert out["focus_paper_ids"] == ["eval_05_wind_turbine"]
    assert out["scope_mode"] == "single_session"
    assert not out["needs_clarification"]


def test_single_paper_discovery_does_not_force_focus():
    intent = RetrievalIntent(
        effective_query="哪篇论文讲了全同态加密？",
        intent="paper_discovery",
        focus_paper_ids=["eval_05_wind_turbine"],
        missing=[],
        confidence=0.9,
    )
    out = validate_intent(intent, intent.effective_query, ["eval_05_wind_turbine"])
    assert out["focus_paper_ids"] == []
    assert out["scope_mode"] == "discovery"
    assert out["retrieval_mode"] == "profile"
    assert not out["needs_clarification"]


def test_constraints_narrow_candidates():
    intent = RetrievalIntent(
        effective_query="最新的量子纠错论文是哪篇？",
        intent="need_clarification",
        focus_paper_ids=[],
        missing=["which_paper"],
        constraints={"time": "latest", "topic": "quantum"},
        confidence=0.9,
    )
    out = validate_intent(intent, intent.effective_query, SESSION)
    assert out["needs_clarification"]
    assert out["candidate_paper_ids"][0] == "eval_10_quantum_error"


def test_missing_which_paper_clarifies():
    intent = RetrievalIntent(
        effective_query="这篇论文摘要说了什么",
        intent="need_clarification",
        focus_paper_ids=[],
        missing=["which_paper"],
        clarification_question="请指明是哪一篇论文。",
        confidence=0.9,
    )
    out = validate_intent(intent, intent.effective_query, SESSION)
    assert out["needs_clarification"]
    assert out["clarification_kind"] == "paper"
    assert out["match_reason"] == "missing_slots"


def test_llm_focus_passes_when_in_session():
    intent = RetrievalIntent(
        effective_query="eval_05 的摘要说了什么？",
        intent="paper_qa",
        focus_paper_ids=["eval_05_wind_turbine"],
        missing=[],
        confidence=0.9,
    )
    out = validate_intent(intent, intent.effective_query, SESSION)
    assert out["focus_paper_ids"] == ["eval_05_wind_turbine"]
    assert not out["needs_clarification"]


def test_missing_what_to_ask_clarifies():
    intent = RetrievalIntent(
        effective_query="eval_10_quantum_error",
        intent="need_clarification",
        focus_paper_ids=[],
        missing=["what_to_ask"],
        confidence=0.9,
    )
    out = validate_intent(intent, intent.effective_query, SESSION)
    assert out["needs_clarification"]
    assert out["clarification_kind"] == "intent"


def test_hallucinated_paper_id_stripped_then_clarify():
    intent = RetrievalIntent(
        effective_query="摘要说了什么",
        intent="paper_qa",
        focus_paper_ids=["not_in_session"],
        missing=[],
        confidence=0.9,
    )
    out = validate_intent(intent, intent.effective_query, SESSION)
    assert out["needs_clarification"]
    assert out["match_reason"] == "no_focus_multi_paper"


def test_low_confidence_clarifies():
    intent = RetrievalIntent(
        effective_query="eval_05 的摘要",
        intent="paper_qa",
        focus_paper_ids=["eval_05_wind_turbine"],
        missing=[],
        confidence=0.5,
    )
    out = validate_intent(intent, intent.effective_query, SESSION)
    assert out["needs_clarification"]
    assert out["match_reason"] == "low_confidence"


def test_discovery_passes_with_profile_mode():
    intent = RetrievalIntent(
        effective_query="哪篇论文讲量子纠错？",
        intent="paper_discovery",
        focus_paper_ids=[],
        missing=[],
        confidence=0.9,
    )
    out = validate_intent(intent, intent.effective_query, SESSION)
    assert not out["needs_clarification"]
    assert out["retrieval_mode"] == "profile"
    assert out["scope_mode"] == "discovery"


if __name__ == "__main__":
    test_single_paper_session_forces_focus_for_qa()
    test_single_paper_discovery_does_not_force_focus()
    test_constraints_narrow_candidates()
    test_missing_which_paper_clarifies()
    test_llm_focus_passes_when_in_session()
    test_missing_what_to_ask_clarifies()
    test_hallucinated_paper_id_stripped_then_clarify()
    test_low_confidence_clarifies()
    test_discovery_passes_with_profile_mode()
    print("All intent_validate tests passed.")
