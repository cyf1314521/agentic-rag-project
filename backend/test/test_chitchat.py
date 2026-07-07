"""方案 B 状态机测试（含 Gateway 委托）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.scope import (
    looks_like_paper_rag,
    scope_check_pre_llm,
    is_meta_not_rag,
)
from agent.turn_resolver import TurnState, resolve_turn
from agent.query_rewrite import is_anaphoric_followup, resolve_effective_query
from agent.intent import RetrievalIntent
from agent.gateway import run_gateway
from agent.session_corpus import PaperGateRepr, SessionCorpus

SESSION = ["eval_03_blockchain_scm", "eval_05_wind_turbine"]


class FakeEmb:
    def embed_query(self, q: str) -> list[float]:
        ql = (q or "").lower()
        if any(k in ql for k in ("比特币", "bitcoin", "区块链", "blockchain")):
            return [1.0, 0.0]
        if any(k in ql for k in ("eval_05", "wind", "tps", "p99")):
            return [0.0, 1.0]
        return [0.1, 0.1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if "blockchain" in t.lower() or "bitcoin" in t.lower():
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out


def _mixed_corpus() -> SessionCorpus:
    return SessionCorpus(
        paper_ids=SESSION,
        reprs=[
            PaperGateRepr("eval_05_wind_turbine", "wind turbine SCADA power TPS"),
            PaperGateRepr(
                "eval_03_blockchain_scm",
                "Bitcoin blockchain peer-to-peer cryptocurrency",
            ),
        ],
        corpus_id="chitchat-test",
    )


def test_greeting_not_anaphoric():
    assert is_meta_not_rag("你好")
    assert not is_anaphoric_followup("你好")


def test_greeting_not_merged():
    msgs = [("human", "eval_05 摘要峰值"), ("ai", "1200 TPS")]
    q, reason = resolve_effective_query("你好", SESSION, msgs)
    assert q == "你好"
    assert reason == ""


def test_gateway_hello():
    d = run_gateway("你好", SESSION, embeddings=FakeEmb(), corpus=_mixed_corpus())
    assert d.action == "out_of_scope"
    assert scope_check_pre_llm("你好", SESSION) == "out_of_scope"


def test_gateway_bitcoin_on_wind_corpus():
    corpus = SessionCorpus(
        paper_ids=["eval_05_wind_turbine"],
        reprs=[PaperGateRepr("eval_05_wind_turbine", "wind turbine only")],
        corpus_id="w",
    )
    d = run_gateway("比特币今天价格多少", ["eval_05_wind_turbine"], embeddings=FakeEmb(), corpus=corpus)
    assert d.action == "out_of_scope"


def test_gateway_bitcoin_on_blockchain_corpus():
    d = run_gateway(
        "比特币共识机制",
        SESSION,
        embeddings=FakeEmb(),
        corpus=_mixed_corpus(),
    )
    assert d.action == "continue"


def test_gateway_anaphoric_p99():
    msgs = [("human", "eval_05 的摘要峰值 TPS？"), ("ai", "1200")]
    d = run_gateway("那 P99 延迟呢？", SESSION, messages=msgs, embeddings=FakeEmb(), corpus=_mixed_corpus())
    assert d.action == "continue"


def test_greeting_turn():
    intent = RetrievalIntent(effective_query="你好", intent="paper_qa", confidence=0.9)
    out = resolve_turn(
        intent, "你好", SESSION, raw_query="你好", scope_continue=False,
    )
    assert out["turn_state"] == TurnState.OUT_OF_SCOPE.value


def test_no_session_turn():
    intent = RetrievalIntent(effective_query="eval_05 摘要", intent="paper_qa", confidence=0.9)
    out = resolve_turn(intent, intent.effective_query, [])
    assert out["turn_state"] == TurnState.UPLOAD_REQUIRED.value


def test_paper_rag_ready():
    intent = RetrievalIntent(
        effective_query="eval_05 的摘要说了什么",
        intent="paper_qa",
        focus_paper_ids=["eval_05_wind_turbine"],
        confidence=0.9,
    )
    out = resolve_turn(intent, intent.effective_query, SESSION, scope_continue=True)
    assert out["turn_state"] == TurnState.RAG_READY.value


def test_multi_paper_clarify():
    intent = RetrievalIntent(
        effective_query="这篇论文摘要说了什么",
        intent="paper_qa",
        focus_paper_ids=[],
        missing=["which_paper"],
        confidence=0.9,
    )
    out = resolve_turn(intent, intent.effective_query, SESSION, scope_continue=True)
    assert out["turn_state"] == TurnState.NEED_CLARIFICATION.value


def test_looks_like_paper_rag():
    assert looks_like_paper_rag("eval_05 摘要说了什么", SESSION)
    assert not looks_like_paper_rag("你好", SESSION)


if __name__ == "__main__":
    test_greeting_not_anaphoric()
    test_greeting_not_merged()
    test_gateway_hello()
    test_gateway_bitcoin_on_wind_corpus()
    test_gateway_bitcoin_on_blockchain_corpus()
    test_gateway_anaphoric_p99()
    test_greeting_turn()
    test_no_session_turn()
    test_paper_rag_ready()
    test_multi_paper_clarify()
    test_looks_like_paper_rag()
    print("All turn resolver tests passed.")
