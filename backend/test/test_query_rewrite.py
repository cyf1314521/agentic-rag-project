"""Unit tests for retrieval-time query rewrite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.query_rewrite import (
    is_anaphoric_followup,
    is_topic_switch,
    merge_anaphoric_followup,
    merge_clarification_followup,
    resolve_effective_query,
)

SESSION = [
    "eval_03_blockchain_scm",
    "eval_05_wind_turbine",
    "eval_10_quantum_error",
]


def test_clarification_paper_id_merge():
    msgs = [
        ("human", "这篇论文的摘要说了什么"),
        ("ai", "当前会话绑定了多篇论文，无法确定您指的是哪一篇。"),
    ]
    q, reason = resolve_effective_query("eval_10_quantum_error", SESSION, msgs)
    assert "eval_10_quantum_error" in q
    assert "摘要" in q
    assert reason == "clarification_followup"


def test_anaphoric_followup_merge():
    msgs = [
        ("human", "eval_05 的摘要峰值吞吐量是多少 TPS？"),
        ("ai", "峰值为 1200 TPS [1]。"),
    ]
    q, reason = resolve_effective_query("那 P99 延迟呢？", SESSION, msgs)
    assert "eval_05" in q or "P99" in q
    assert "追问" in q or "P99" in q
    assert reason == "anaphoric_followup"


def test_full_question_unchanged():
    q = "eval_05 的摘要说了什么？"
    out, reason = resolve_effective_query(q, SESSION, [])
    assert out == q
    assert reason == ""


def test_is_anaphoric():
    assert is_anaphoric_followup("那 P99 呢？")
    assert not is_anaphoric_followup("eval_05 的摘要说了什么？")


def test_clarification_via_pending_query_without_messages():
    q, reason = resolve_effective_query(
        "eval_10_quantum_error",
        SESSION,
        [],
        pending_user_query="这篇论文的摘要说了什么",
    )
    assert "eval_10_quantum_error" in q
    assert "摘要" in q
    assert reason == "clarification_followup"


def test_paper_id_only_without_context_unchanged():
    q, reason = resolve_effective_query("eval_10_quantum_error", SESSION, [])
    assert q == "eval_10_quantum_error"
    assert reason == ""


def test_topic_switch_skips_pending_merge():
    q, reason = resolve_effective_query(
        "帮我写一个 Python 脚本统计 CSV",
        SESSION,
        [],
        pending_user_query="这篇论文的摘要说了什么",
    )
    assert q == "帮我写一个 Python 脚本统计 CSV"
    assert reason == "topic_switch"


def test_topic_switch_short_dismiss():
    assert is_topic_switch("算了", pending_user_query="这篇论文说了什么")
    assert not is_anaphoric_followup("算了")


if __name__ == "__main__":
    test_clarification_paper_id_merge()
    test_anaphoric_followup_merge()
    test_full_question_unchanged()
    test_is_anaphoric()
    test_clarification_via_pending_query_without_messages()
    test_paper_id_only_without_context_unchanged()
    test_topic_switch_skips_pending_merge()
    test_topic_switch_short_dismiss()
    print("All query_rewrite tests passed.")
