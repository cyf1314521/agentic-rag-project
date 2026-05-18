"""
LangGraph / LangChain 工具定义。

paper_retrieval：可被 ReAct 式 Agent 调用的论文检索工具。
ContextVar query_type：在 classify 节点设置后，工具内可读取以做章节类型过滤。

注意：当前主流程通过 dependencies.RetrieverTool 直接检索，
本模块工具主要用于测试脚本或未来扩展 tool-calling Agent。
"""

import sys
from pathlib import Path
from typing import Optional
from contextvars import ContextVar

from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from rag.retrieval import Retriever
from rag.citation import CitationExtractor

_retriever: Optional[Retriever] = None
# 异步/多请求场景下传递当前请求的 query_type（由 classify_query 写入）
_query_type_ctx: ContextVar[str] = ContextVar("query_type", default="general")

# 查询类型 → Milvus section_type 过滤条件（与 nodes.retrieve 中一致）
_ROUTE_CONFIG: dict[str, list[str] | None] = {
    "experimental_result": ["experiment"],
    "method": ["method"],
    "background": ["background"],
    "general": None,
}


def get_retriever() -> Retriever:
    """懒加载全局 Retriever（独立脚本用，与 app.dependencies 单例分离）。"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(
            embedding_model=Config.EMBEDDING_MODEL,
            reranker_model=Config.RERANKER_MODEL,
            milvus_uri=Config.MILVUS_URI,
            collection_name=Config.COLLECTION_NAME,
            enable_cache=Config.ENABLE_CACHE,
        )
    return _retriever


def set_query_type(query_type: str):
    """由 classify_query 节点调用，供后续 paper_retrieval 读取。"""
    _query_type_ctx.set(query_type)


@tool
def paper_retrieval(query: str) -> str:
    """
    检索学术论文知识库，返回相关片段及来源信息。

    当需要从已索引论文中查找事实以回答研究问题时使用。
    返回带 [序号] 的段落及 Paper/Section/Page 元数据。
    """
    retriever = get_retriever()
    query_type = _query_type_ctx.get()
    section_type_filter = _ROUTE_CONFIG.get(query_type)

    docs = retriever.retrieve(
        query=query,
        k=Config.TOP_K,
        use_hyde=False,
        rerank=True,
        expand_parent=True,
        rrf_k=Config.RRF_K,
        fetch_k=Config.FETCH_K,
        section_type_filter=section_type_filter,
    )

    if not docs:
        return "No relevant documents found for the given query."

    citations = CitationExtractor.extract_all(docs)
    parts = []
    for i, (doc, cite) in enumerate(zip(docs, citations), 1):
        source = CitationExtractor.format_citation(cite)
        parts.append(f"[{i}] {doc.page_content}\n    Source: {source}")

    return "\n\n".join(parts)
