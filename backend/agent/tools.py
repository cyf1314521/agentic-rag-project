"""
Agent 运行时上下文：查询类型 → Milvus section_type 过滤。

classify_query 节点写入 ContextVar；retrieve 节点读取 SECTION_TYPE_ROUTE。
"""

from contextvars import ContextVar

# 查询类型 → Milvus section_type 过滤（retrieve 节点使用）
SECTION_TYPE_ROUTE: dict[str, list[str] | None] = {
    "experimental_result": ["experiment"],
    "method": ["method"],
    "background": ["background"],
    "general": None,
}

_query_type_ctx: ContextVar[str] = ContextVar("query_type", default="general")


def set_query_type(query_type: str) -> None:
    """由 classify_query 节点调用，供 retrieve 读取。"""
    _query_type_ctx.set(query_type)


def get_query_type() -> str:
    return _query_type_ctx.get()
