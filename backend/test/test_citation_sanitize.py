"""Unit tests for citation sanitization after synthesis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.citation import sanitize_answer_citations


def test_strips_out_of_range_citation():
    answer = "峰值吞吐量为 1200 TPS [1]，延迟见 [99]。"
    cleaned, stripped = sanitize_answer_citations(answer, 3)
    assert "[99]" not in cleaned
    assert "[1]" in cleaned
    assert 99 in stripped


def test_keeps_valid_multi_cite():
    answer = "对比结果 [1, 2] 显示一致。"
    cleaned, stripped = sanitize_answer_citations(answer, 2)
    assert "[1, 2]" in cleaned or "[1]" in cleaned
    assert not stripped


def test_no_citations_when_max_zero():
    answer = "无引用答案 [1]"
    cleaned, stripped = sanitize_answer_citations(answer, 0)
    assert "[1]" not in cleaned
    assert stripped == [1]


if __name__ == "__main__":
    test_strips_out_of_range_citation()
    test_keeps_valid_multi_cite()
    test_no_citations_when_max_zero()
    print("All citation sanitize tests passed.")
