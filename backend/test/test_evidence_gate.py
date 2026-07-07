"""Evidence gate 单元测试：rerank / fallback 熔断。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from rag.evidence_gate import evaluate_evidence


def _doc(**meta) -> Document:
    return Document(page_content="text", metadata=meta)


def test_empty_docs_fails():
    v = evaluate_evidence([])
    assert not v.passed
    assert v.reason == "empty_docs"
    assert v.signal == "none"


def test_rerank_above_threshold_passes():
    docs = [_doc(rerank_score=0.42), _doc(rerank_score=0.12)]
    v = evaluate_evidence(docs, min_rerank=0.15)
    assert v.passed
    assert v.signal == "rerank"
    assert v.top_score == 0.42


def test_rerank_below_threshold_fails():
    docs = [_doc(rerank_score=0.05)]
    v = evaluate_evidence(docs, min_rerank=0.15)
    assert not v.passed
    assert v.reason == "rerank_below_threshold"


def test_fallback_dense_when_no_rerank():
    docs = [_doc(score=0.12), _doc(score=0.04)]
    v = evaluate_evidence(docs, min_fallback=0.08)
    assert v.passed
    assert v.signal == "dense"
    assert v.top_score == 0.12


def test_fallback_sparse_below_threshold():
    docs = [_doc(sparse_score=0.03)]
    v = evaluate_evidence(docs, min_fallback=0.08)
    assert not v.passed
    assert v.signal == "bm25"
    assert v.reason == "fallback_below_threshold"


def test_no_score_metadata_fails():
    docs = [_doc(chunk_id="x")]
    v = evaluate_evidence(docs)
    assert not v.passed
    assert v.reason == "no_score_metadata"
