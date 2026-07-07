"""Corpus Gate 与 Gateway 单元测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.content_gate import bm25_lite_score, build_gate_query, score_content_relevance
from agent.gateway import run_gateway
from agent.session_corpus import PaperGateRepr, SessionCorpus, invalidate_corpus_cache
from agent.scope import is_meta_not_rag

SESSION_WIND = ["eval_05_wind_turbine"]
SESSION_MIX = ["eval_03_blockchain_scm", "eval_05_wind_turbine"]


class FakeEmb:
    """按 query 关键词返回不同方向向量，便于断言 cosine。"""

    def embed_query(self, q: str) -> list[float]:
        ql = (q or "").lower()
        if any(k in ql for k in ("比特币", "bitcoin", "区块链", "blockchain", "共识")):
            return [1.0, 0.0, 0.0]
        if any(k in ql for k in ("风力", "wind", "turbine", "tps", "p99")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def _corpus(wind_text: str, chain_text: str) -> SessionCorpus:
    return SessionCorpus(
        paper_ids=list(SESSION_MIX),
        reprs=[
            PaperGateRepr("eval_05_wind_turbine", wind_text),
            PaperGateRepr("eval_03_blockchain_scm", chain_text),
        ],
        corpus_id="test",
    )


def test_bm25_lite_overlap():
    assert bm25_lite_score("bitcoin consensus", "Paper about Bitcoin consensus mechanism") > 0.3
    assert bm25_lite_score("wind turbine", "bitcoin blockchain") == 0.0


def test_build_gate_query_anaphoric():
    msgs = [
        ("human", "eval_05 的摘要峰值 TPS？"),
        ("ai", "1200 TPS"),
    ]
    gq = build_gate_query("那 P99 延迟呢？", messages=msgs)
    assert "eval_05" in gq or "P99" in gq


def test_corpus_gate_bitcoin_on_wind_session():
    corpus = SessionCorpus(
        paper_ids=SESSION_WIND,
        reprs=[PaperGateRepr("eval_05_wind_turbine", "wind turbine SCADA power generation")],
        corpus_id="t1",
    )
    emb = FakeEmb()
    result = score_content_relevance("比特币今天价格", corpus, emb)
    assert result.score < 0.15


def test_corpus_gate_bitcoin_on_blockchain_session():
    corpus = _corpus(
        "wind turbine SCADA",
        "Bitcoin blockchain peer-to-peer electronic cash consensus",
    )
    emb = FakeEmb()
    result = score_content_relevance("比特币的共识机制", corpus, emb)
    assert result.score >= 0.15
    assert result.best_paper_id == "eval_03_blockchain_scm"


def test_gateway_meta_hello():
    invalidate_corpus_cache()
    d = run_gateway("你好", SESSION_MIX, embeddings=FakeEmb())
    assert d.action == "out_of_scope"
    assert d.reason == "meta"
    assert is_meta_not_rag("你好")


def test_gateway_looks_like_fastpath():
    d = run_gateway("eval_05 摘要说了什么", SESSION_MIX, embeddings=FakeEmb())
    assert d.action == "continue"
    assert d.reason == "looks_like_fastpath"


def test_gateway_bitcoin_blocked_on_wind():
    corpus = SessionCorpus(
        paper_ids=SESSION_WIND,
        reprs=[PaperGateRepr("eval_05_wind_turbine", "wind turbine SCADA")],
        corpus_id="t2",
    )
    d = run_gateway(
        "比特币今天价格多少",
        SESSION_WIND,
        embeddings=FakeEmb(),
        corpus=corpus,
    )
    assert d.action == "out_of_scope"
    assert d.reason in ("low_relevance", "task_off_domain")


def test_gateway_bitcoin_continue_on_blockchain():
    corpus = _corpus(
        "wind only",
        "Bitcoin blockchain cryptocurrency consensus Nakamoto",
    )
    d = run_gateway(
        "比特币共识机制是什么",
        SESSION_MIX,
        embeddings=FakeEmb(),
        corpus=corpus,
    )
    assert d.action == "continue"
    assert d.reason in ("content_ok", "content_high", "looks_like_fastpath")


def test_gateway_anaphoric_p99_continue():
    corpus = _corpus(
        "wind turbine peak TPS throughput SCADA",
        "blockchain",
    )
    msgs = [("human", "eval_05 的摘要峰值 TPS？"), ("ai", "1200")]
    d = run_gateway(
        "那 P99 延迟呢？",
        SESSION_MIX,
        messages=msgs,
        embeddings=FakeEmb(),
        corpus=corpus,
    )
    assert d.action == "continue"


if __name__ == "__main__":
    test_bm25_lite_overlap()
    test_build_gate_query_anaphoric()
    test_corpus_gate_bitcoin_on_wind_session()
    test_corpus_gate_bitcoin_on_blockchain_session()
    test_gateway_meta_hello()
    test_gateway_looks_like_fastpath()
    test_gateway_bitcoin_blocked_on_wind()
    test_gateway_bitcoin_continue_on_blockchain()
    test_gateway_anaphoric_p99_continue()
    print("All content_gate tests passed.")
