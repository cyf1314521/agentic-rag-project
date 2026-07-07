"""Unit tests for multi-PDF paper scope resolution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.paper_scope import resolve_paper_scope, effective_retrieval_paper_ids


SESSION = [
    "eval_03_blockchain_scm",
    "eval_05_wind_turbine",
    "eval_10_quantum_error",
]


def test_single_paper_session_auto_focus():
    r = resolve_paper_scope("摘要说了什么？", ["eval_05_wind_turbine"])
    assert r["scope_mode"] == "single_session"
    assert r["focus_paper_ids"] == ["eval_05_wind_turbine"]
    assert not r["needs_clarification"]


def test_ambiguous_bare_abstract():
    r = resolve_paper_scope("摘要说了什么？", SESSION)
    assert r["needs_clarification"]
    assert r["scope_mode"] == "ambiguous"
    assert r["focus_paper_ids"] == []
    assert "eval_05" in r["clarification_message"]


def test_explicit_eval_number():
    r = resolve_paper_scope("eval_05 的摘要说了什么？", SESSION)
    assert r["scope_mode"] == "explicit"
    assert r["focus_paper_ids"] == ["eval_05_wind_turbine"]
    assert not r["needs_clarification"]


def test_explicit_paper_id_substring():
    r = resolve_paper_scope("eval_10_quantum_error 里提到了什么纠错码？", SESSION)
    assert r["focus_paper_ids"] == ["eval_10_quantum_error"]


def test_session_wide_compare():
    r = resolve_paper_scope("对比 eval_03 和 eval_05 的摘要", SESSION)
    assert r["scope_mode"] == "session_wide"
    assert not r["needs_clarification"]


def test_effective_retrieval_ids():
    assert effective_retrieval_paper_ids(SESSION, ["eval_05_wind_turbine"]) == ["eval_05_wind_turbine"]
    assert effective_retrieval_paper_ids(SESSION, []) == SESSION
    assert effective_retrieval_paper_ids([], []) is None


def test_topic_anchor_not_ambiguous():
    r = resolve_paper_scope(
        "这篇关于薄膜太阳能电池的论文，摘要里报告的最高光电转换效率是多少？",
        SESSION + ["eval_01_solar_pv"],
    )
    assert not r["needs_clarification"]
    assert r["scope_mode"] == "explicit"
    assert r["focus_paper_ids"] == ["eval_01_solar_pv"]
    assert r["match_reason"] == "topic_anchor"


def test_eval_topic_focus_solar():
    session = ["eval_01_solar_pv", "eval_02_ocean_ph", "eval_03_blockchain_scm"]
    r = resolve_paper_scope(
        "该论文摘要的研究对象是什么类型的太阳能电池？",
        session,
    )
    assert not r["needs_clarification"]
    assert r["focus_paper_ids"] == ["eval_01_solar_pv"]


def test_ocean_topic_focus():
    session = ["eval_01_solar_pv", "eval_02_ocean_ph", "eval_03_blockchain_scm"]
    r = resolve_paper_scope(
        "这篇海洋酸化论文的摘要预测到 2050 年北太平洋表层 pH 是多少？",
        session,
    )
    assert r["focus_paper_ids"] == ["eval_02_ocean_ph"]


def test_deictic_short_topic_ambiguous():
    r = resolve_paper_scope("该论文摘要说了什么？", SESSION)
    assert r["needs_clarification"]


if __name__ == "__main__":
    test_single_paper_session_auto_focus()
    test_ambiguous_bare_abstract()
    test_explicit_eval_number()
    test_explicit_paper_id_substring()
    test_session_wide_compare()
    test_effective_retrieval_ids()
    test_topic_anchor_not_ambiguous()
    test_eval_topic_focus_solar()
    test_ocean_topic_focus()
    test_deictic_short_topic_ambiguous()
    print("All paper_scope tests passed.")
