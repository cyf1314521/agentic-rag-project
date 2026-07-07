"""父块扩展时子块检索分数应保留在 metadata 中，供 evidence_gate 使用。"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from rag.retrieval import Retriever


def test_expand_to_parents_propagates_rerank_score():
    retriever = Retriever.__new__(Retriever)
    retriever._parent_store = MagicMock()

    child = Document(
        page_content="abstract snippet",
        metadata={
            "chunk_parent_id": "parent-1",
            "rerank_score": 0.55,
            "chunk_id": "child-1",
        },
    )
    parent_hit = Document(
        page_content="full parent with abstract",
        metadata={"chunk_id": "parent-1"},
    )
    retriever._parent_store.similarity_search.return_value = [parent_hit]

    out = retriever._expand_to_parents([child])

    assert len(out) == 1
    assert out[0].metadata.get("rerank_score") == 0.55
    assert out[0].page_content == "full parent with abstract"


def test_expand_to_parents_keeps_best_score_across_children():
    retriever = Retriever.__new__(Retriever)
    retriever._parent_store = MagicMock()

    children = [
        Document(
            page_content="weak hit",
            metadata={"chunk_parent_id": "p1", "rerank_score": 0.2, "chunk_id": "c1"},
        ),
        Document(
            page_content="strong hit",
            metadata={"chunk_parent_id": "p1", "rerank_score": 0.71, "chunk_id": "c2"},
        ),
    ]
    parent_hit = Document(page_content="parent", metadata={"chunk_id": "p1"})
    retriever._parent_store.similarity_search.return_value = [parent_hit]

    out = retriever._expand_to_parents(children)

    assert out[0].metadata.get("rerank_score") == 0.71
