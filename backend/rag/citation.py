"""
引用元数据提取与格式化。

从 LangChain Document.metadata 抽取 paper_id、章节路径、页码等，
供前端展示及生成答案中的 [Source: ...] 标注。
"""

import re
from typing import Dict, List

from langchain_core.documents import Document

_CITATION_BRACKET_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")


def parse_citation_indices(answer: str) -> list[int]:
    """从答案解析 [1]、[1, 2]、[1][2] 等 1-based 引用编号。"""
    indices: list[int] = []
    seen: set[int] = set()
    for m in re.finditer(r"\[([^\]]+)\]", answer or ""):
        for part in re.split(r"[,，\s]+", m.group(1).strip()):
            if part.isdigit():
                n = int(part)
                if n >= 1 and n not in seen:
                    seen.add(n)
                    indices.append(n)
    return indices


def is_citation_only_answer(text: str) -> bool:
    """True when answer is empty or only bracket indices like [1] / [1], [2]."""
    t = (text or "").strip()
    if not t:
        return True
    without_cites = _CITATION_BRACKET_RE.sub("", t).strip()
    without_cites = re.sub(r"^[,，\s\.。；;]+|[,，\s\.。；;]+$", "", without_cites)
    if not without_cites:
        return True
    if re.search(r"[\d\u4e00-\u9fff]", without_cites):
        return False
    return len(without_cites) < 12


def sanitize_answer_citations(
    answer: str,
    max_index: int,
) -> tuple[str, list[int]]:
    """
    剥离合成答案中超出 Citation Index 范围的 [n]（防小模型引用漂移）。

    Returns:
        (cleaned_answer, stripped_invalid_indices)
    """
    if not answer or max_index <= 0:
        stripped = parse_citation_indices(answer)
        if stripped:
            return re.sub(r"\[\d+(?:,\s*\d+)*\]", "", answer).strip(), stripped
        return answer, []

    stripped: list[int] = []

    def _replace(m: re.Match[str]) -> str:
        valid: list[str] = []
        for part in re.split(r"[,，\s]+", m.group(1).strip()):
            if not part.isdigit():
                continue
            n = int(part)
            if 1 <= n <= max_index:
                valid.append(str(n))
            else:
                stripped.append(n)
        if not valid:
            return ""
        if len(valid) == 1:
            return f"[{valid[0]}]"
        return "[" + ", ".join(valid) + "]"

    cleaned = re.sub(r"\[([^\]]+)\]", _replace, answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;，。；])", r"\1", cleaned)
    return cleaned.strip(), stripped


def paper_ids_from_citations(answer: str, citations: list[dict]) -> list[str]:
    """答案 [n] 对应 citations[n-1] 的 paper_id（去重排序）。"""
    paper_ids: set[str] = set()
    for n in parse_citation_indices(answer):
        if 1 <= n <= len(citations):
            pid = citations[n - 1].get("paper_id")
            if pid:
                paper_ids.add(str(pid))
    return sorted(paper_ids)


def scope_ok_for_paper(paper_ids_cited: list[str], expected: str) -> bool:
    return len(paper_ids_cited) == 1 and paper_ids_cited[0] == expected


class CitationExtractor:
    """检索结果 → 结构化引用 → 人类可读字符串。"""

    @staticmethod
    def extract_citation(doc: Document) -> Dict:
        """从单条 Document 的 metadata 提取引用字段。"""
        meta = doc.metadata
        return {
            "paper_id": meta.get("paper_id", ""),
            "section": meta.get("section_path", ""),
            "page": meta.get("page_num", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "node_type": meta.get("node_type", ""),
            # 保留完整 metadata 供 VLM（image_path 等）
            "metadata": meta,
        }

    @staticmethod
    def format_citation(citation: Dict) -> str:
        """格式化为 'Paper: x | Section: y | Page: z'。"""
        parts = []
        if citation["paper_id"]:
            parts.append(f"Paper: {citation['paper_id']}")
        if citation["section"]:
            parts.append(f"Section: {citation['section']}")
        if citation["page"]:
            parts.append(f"Page: {citation['page']}")
        return " | ".join(parts) if parts else "Unknown source"

    @staticmethod
    def extract_all(docs: List[Document]) -> List[Dict]:
        """批量提取，与 docs 顺序一一对应。"""
        return [CitationExtractor.extract_citation(doc) for doc in docs]
