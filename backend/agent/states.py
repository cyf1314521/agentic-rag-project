"""
LangGraph 状态定义。

包含：
- 顶层 AgentState：主图（summarize → classify → analyze → 并行 sub_agent → synthesize）
- 子图 SubAgentState：单个子查询的 retrieve → generate → reflect 循环
- 自定义 reducer：并行子 Agent 返回的 sub_answers / citations 合并策略
"""

import operator
from typing import Annotated, TypedDict, cast

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.types import Overwrite


class SubAnswer(TypedDict):
    """单个子 Agent 完成后的结构化输出，供最终合成使用。"""
    query: str       # 子查询原文
    answer: str      # 该子查询下的生成答案（含 [n] 引用）
    citations: list[dict]  # 引用元数据列表
    status: str      # "ok" | "failed"
    error: str       # 失败原因；成功时为 ""


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
    trace_id: str                 # 可观测日志关联 ID（chat 入口生成）
    turn_id: str                  # 本轮请求 ID（幂等 / 续跑预留）
    query_type: str               # classify 节点输出：experimental_result | method | background | general
    paper_ids: list[str]         # 本会话检索范围（session_papers）
    summary: str                  # 超长对话的 LLM 摘要
    documents: list[str]          # 预留字段
    query_complexity: str         # analyze：simple | complex
    sub_queries: list[str]        # analyze 节点分解的子查询列表
    sub_answers: Annotated[list[SubAnswer], merge_sub_answers]  # 各子 Agent 答案
    answer: str                   # synthesize 后的最终回答
    citations: Annotated[list[dict], merge_citations]  # 合成阶段汇总的引用
    synth_messages: list[AnyMessage]  # prepare_synthesis 构建的 LLM 消息列表
    failed_sub_queries: list[dict]  # prepare_synthesis：硬失败的子问题 [{query, error}]


def fresh_turn_state(
    query: str,
    trace_id: str,
    paper_ids: list[str] | None = None,
    turn_id: str = "",
) -> AgentState:
    """
    每轮对话/评测的初始主图 state。

    使用 Overwrite 清空带 reducer 的列表字段；LangGraph 输入类型与 TypedDict 不完全一致，故 cast。
    """
    return cast(
        AgentState,
        {
            "query": query,
            "trace_id": trace_id,
            "turn_id": turn_id,
            "paper_ids": paper_ids or [],
            "messages": [],
            "summary": "",
            "query_type": "general",
            "query_complexity": "simple",
            "documents": [],
            "sub_queries": Overwrite([]),
            "sub_answers": Overwrite([]),
            "citations": Overwrite([]),
            "answer": "",
            "synth_messages": [],
            "failed_sub_queries": [],
        },
    )


class SubAgentState(TypedDict):
    """
    子图状态：处理单个子查询的 RAG 循环。
    """
    query: str
    trace_id: str                 # 继承自主图，用于可观测日志
    query_type: str               # 继承自主图，用于检索 section_type 路由
    paper_ids: list[str]         # 继承自主图，Milvus paper_id 过滤
    documents: list[str]          # 检索到的上下文（已格式化为带 Source 的字符串）
    answer: str
    citations: list[dict]
    is_sufficient: bool           # reflect 节点判定是否可结束
    retry_queries: Annotated[list[str], operator.add]  # 不足时追加的补充检索 query
    retries: int                  # 已重试次数
    needs_vlm: bool               # 是否已在生成阶段使用/标记 VLM
