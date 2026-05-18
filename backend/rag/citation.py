"""
引用元数据提取与格式化。

从 LangChain Document.metadata 抽取 paper_id、章节路径、页码等，
供前端展示及生成答案中的 [Source: ...] 标注。
"""

from typing import Dict, List
from langchain_core.documents import Document


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
