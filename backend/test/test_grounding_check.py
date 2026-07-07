"""grounding_check 分句与覆盖率测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.grounding_check import check_grounding, _split_sentences


def test_split_sentences_protects_decimal_percent():
    text = "实验结果表明，该模型的准确率达到了 94.19%，优于基线方法。"
    parts = _split_sentences(text)
    assert len(parts) == 1
    assert "94.19%" in parts[0]


def test_grounding_passes_multi_sentence_with_trailing_citation():
    answer = (
        "摘 要：花卉分类是重要基础工作，传统方法效率低。"
        "针对这些挑战，本文提出 CNN 方法。"
        "实验结果表明准确率达到 94.19% [5]。"
    )
    citations = [{"chunk_id": f"c{i}"} for i in range(5)]
    verdict = check_grounding(answer, citations, min_citation_ratio=0.2)
    assert verdict.ok
    assert verdict.reason == "ok"


def test_grounding_fails_when_no_brackets():
    answer = "摘 要：花卉分类是重要基础工作。"
    citations = [{"chunk_id": "c1"}]
    verdict = check_grounding(answer, citations, min_citation_ratio=0.2)
    assert not verdict.ok
    assert verdict.reason == "citations_but_no_brackets"


if __name__ == "__main__":
    test_split_sentences_protects_decimal_percent()
    test_grounding_passes_multi_sentence_with_trailing_citation()
    test_grounding_fails_when_no_brackets()
    print("ok")
