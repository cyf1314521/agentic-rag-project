"""
合成答案 groundedness 轻量校验（引用覆盖率）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import Config
from .citation import parse_citation_indices, sanitize_answer_citations

_CITATION_BRACKET_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")
_DECIMAL_DOT = "__DECIMAL_DOT__"


def _split_sentences(text: str) -> list[str]:
    """分句；保护 94.19% 等小数点，避免误切成多句拉低引文覆盖率。"""
    guarded = re.sub(r"(\d)\.(\d)", rf"\1{_DECIMAL_DOT}\2", text or "")
    parts = [s.strip() for s in re.split(r"[。！？!?]+", guarded) if s.strip()]
    if not parts:
        parts = [guarded.strip()] if guarded.strip() else []
    restored = [p.replace(_DECIMAL_DOT, ".") for p in parts]
    return restored or [text.strip()]


@dataclass
class GroundingVerdict:
    ok: bool
    citation_coverage: float
    orphan_citations: list[int] = field(default_factory=list)
    stripped_indices: list[int] = field(default_factory=list)
    reason: str = "ok"


def check_grounding(
    answer: str,
    citations: list[dict],
    *,
    min_citation_ratio: float | None = None,
) -> GroundingVerdict:
    if not Config.GROUNDING_CHECK_ENABLED:
        return GroundingVerdict(ok=True, citation_coverage=1.0)

    min_ratio = (
        min_citation_ratio
        if min_citation_ratio is not None
        else Config.GROUNDING_MIN_CITATION_RATIO
    )
    text = (answer or "").strip()
    n_cites = len(citations or [])

    if not text:
        return GroundingVerdict(ok=False, citation_coverage=0.0, reason="empty_answer")

    cleaned, stripped = sanitize_answer_citations(text, n_cites)
    indices = parse_citation_indices(cleaned)

    if n_cites > 0 and not indices:
        return GroundingVerdict(
            ok=False,
            citation_coverage=0.0,
            stripped_indices=stripped,
            reason="citations_but_no_brackets",
        )

    sentences = _split_sentences(cleaned)

    cited_sentences = sum(
        1 for s in sentences if _CITATION_BRACKET_RE.search(s)
    )
    coverage = cited_sentences / len(sentences) if sentences else 0.0

    # 长段落（如中文摘要）常在末尾集中标注一次 [n]；有有效引用即可，不必每句都标。
    if indices and cited_sentences >= 1:
        return GroundingVerdict(
            ok=True,
            citation_coverage=coverage,
            stripped_indices=stripped,
            reason="ok",
        )

    if n_cites > 0 and coverage < min_ratio and len(cleaned) > 80:
        return GroundingVerdict(
            ok=False,
            citation_coverage=coverage,
            stripped_indices=stripped,
            reason="low_citation_coverage",
        )

    if stripped and len(stripped) > max(2, len(indices)):
        return GroundingVerdict(
            ok=False,
            citation_coverage=coverage,
            stripped_indices=stripped,
            reason="many_invalid_citations",
        )

    return GroundingVerdict(
        ok=True,
        citation_coverage=coverage,
        stripped_indices=stripped,
        reason="ok",
    )
