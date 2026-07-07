"""Unit tests for paper profile collection and formatting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.models import PaperNode
from rag.paper_profile import collect_profile_source, _format_retrieval_text, _fallback_profile


def test_collect_profile_source():
    nodes = [
        PaperNode(
            node_id="1",
            paper_id="eval_10",
            node_type="section_header",
            text="Abstract",
            page_num=1,
            order=1,
            section_path=["Abstract"],
        ),
        PaperNode(
            node_id="2",
            paper_id="eval_10",
            node_type="paragraph",
            text="摘要：本文研究量子纠错表面码。",
            page_num=1,
            order=2,
            section_path=["Abstract"],
        ),
    ]
    blob = collect_profile_source(nodes, max_chars=5000)
    assert "Abstract" in blob
    assert "量子纠错" in blob


def test_format_retrieval_text():
    meta = _fallback_profile("eval_10_quantum_error", "quantum error correction paper")
    text = _format_retrieval_text("eval_10_quantum_error", meta)
    assert "eval_10_quantum_error" in text
    assert "Paper:" in text


if __name__ == "__main__":
    test_collect_profile_source()
    test_format_retrieval_text()
    print("All paper_profile tests passed.")
