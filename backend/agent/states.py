"""
LangGraph 状态定义。

包含：
- 顶层 AgentState：主图（summarize → classify → analyze → 并行 sub_agent → synthesize）
- 子图 SubAgentState：单个子查询的 retrieve → generate → reflect 循环
- 自定义 reducer：并行子 Agent 返回的 sub_answers / citations 合并策略
"""

import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SubAnswer(TypedDict):
    """单个子 Agent 完成后的结构化输出，供最终合成使用。"""
    query: str       # 子查询原文
    answer: str      # 该子查询下的生成答案（含 [n] 引用）
    citations: list[dict]  # 引用元数据列表


def merge_sub_answers(left: list[SubAnswer], right: list[SubAnswer]) -> list[SubAnswer]:
    """
    合并并行 sub_agent 节点返回的 sub_answers。

    同一 query 以最新结果覆盖；不同 query 追加到列表。
    LangGraph 在多个 Send 分支汇聚时调用此 reducer。
    """
    existing = {a["query"] for a in left}
    merged = list(left)
    for a in right:
        if a["query"] in existing:
            merged = [a if item["query"] == a["query"] else item for item in merged]
        else:
            merged.append(a)
    return merged


def merge_citations(left: list[dict], right: list[dict]) -> list[dict]:
    """按 chunk_id 去重合并引用列表。"""
    seen = set()
    merged = []
    for c in left + right:
        key = c.get("chunk_id") or id(c)
        if key not in seen:
            seen.add(key)
            merged.append(c)
    return merged


class AgentState(TypedDict):
    """
    主图全局状态。

    messages 使用 add_messages reducer，支持追加与 RemoveMessage 删除旧消息。
    """
    messages: Annotated[list[AnyMessage], add_messages]  # 多轮对话历史（Human/AI）
    query: str                    # 当前轮用户问题
    query_type: str               # classify 节点输出：experimental_result | method | background | general
    summary: str                  # 超长对话的 LLM 摘要
    documents: list[str]          # 预留字段
    sub_queries: list[str]        # analyze 节点分解的子查询列表
    sub_answers: Annotated[list[SubAnswer], merge_sub_answers]  # 各子 Agent 答案
    answer: str                   # synthesize 后的最终回答
    citations: Annotated[list[dict], merge_citations]  # 合成阶段汇总的引用
    synth_messages: list[AnyMessage]  # prepare_synthesis 构建的 LLM 消息列表


class SubAgentState(TypedDict):
    """
    子图状态：处理单个子查询的 RAG 循环。
    """
    query: str
    query_type: str               # 继承自主图，用于检索 section_type 路由
    documents: list[str]          # 检索到的上下文（已格式化为带 Source 的字符串）
    answer: str
    citations: list[dict]
    is_sufficient: bool           # reflect 节点判定是否可结束
    retry_queries: Annotated[list[str], operator.add]  # 不足时追加的补充检索 query
    retries: int                  # 已重试次数
    needs_vlm: bool               # 是否已在生成阶段使用/标记 VLM
