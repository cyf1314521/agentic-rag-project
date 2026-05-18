"""
LangGraph 多智能体图组装。

主图：summarize → classify → analyze → [Send 并行 sub_agent] → prepare_synthesis → synthesize
子图（每个 sub_query 一份）：retrieve → generate → reflect → (retry | END)
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Send

from .states import AgentState, SubAgentState, SubAnswer
from .nodes import (
    analyze_query,
    classify_query,
    prepare_synthesis,
    synthesize,
    summarize_conversation,
    retrieve,
    generate,
    reflect,
    should_retry,
    prepare_retry,
)


def _collect_sub_answer(state: SubAgentState) -> dict:
    """子图结束后，将结果包装为 sub_answers 列表项供主图 reducer 合并。"""
    return {
        "sub_answers": [SubAnswer(
            query=state["query"],
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
        )]
    }


def _build_sub_agent_graph(
    llm: BaseChatModel,
    retriever,
    citation_extractor,
    max_retries: int,
    vision_service=None,
) -> StateGraph:
    """
    构建单个子 Agent 的状态图（未 compile）。

    反思不足且未超 max_retries 时走 prepare_retry → retrieve 再次检索。
    """
    sg = StateGraph(SubAgentState)

    async def retrieve_node(state: SubAgentState) -> dict:
        return await retrieve(state, retriever=retriever, citation_extractor=citation_extractor)

    async def generate_node(state: SubAgentState) -> dict:
        return await generate(state, llm=llm, vision_service=vision_service)

    async def reflect_node(state: SubAgentState) -> dict:
        return await reflect(state, llm=llm, vision_service=vision_service)

    def retry_router(state: SubAgentState) -> str:
        return should_retry(state, max_retries=max_retries)

    sg.add_node("retrieve", retrieve_node)
    sg.add_node("generate", generate_node)
    sg.add_node("reflect", reflect_node)
    sg.add_node("prepare_retry", prepare_retry)

    sg.add_edge(START, "retrieve")
    sg.add_edge("retrieve", "generate")
    sg.add_edge("generate", "reflect")
    sg.add_conditional_edges("reflect", retry_router, {
        "retry": "prepare_retry",
        "done": END,
    })
    sg.add_edge("prepare_retry", "retrieve")

    return sg


def build_graph(
    llm: BaseChatModel,
    retriever,
    citation_extractor,
    max_retries: int = 2,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    vision_service=None,
):
    """
    构建并 compile 完整主图。

    Args:
        llm: 主语言模型
        retriever: RetrieverTool 或兼容 invoke(query, section_type_filter) 的对象
        citation_extractor: 引用提取类（通常为 CitationExtractor）
        max_retries: 子 Agent 反思重试上限
        checkpointer: Postgres/Memory saver，用于多轮 thread 状态
        vision_service: 可选 VisionService，用于图表 VLM
    """
    def dispatch(state: AgentState):
        """analyze 之后：为每个 sub_query 创建一个 Send，并行执行 sub_agent。"""
        return [Send("sub_agent", {"query": q, "query_type": state.get("query_type", "general")}) for q in state["sub_queries"]]

    async def sub_agent_node(state: dict) -> dict:
        """每个 Send  payload 进入此节点：编译子图并 ainvoke。"""
        sub_graph = _build_sub_agent_graph(llm, retriever, citation_extractor, max_retries, vision_service).compile()
        sub_input = {
            "query": state["query"],
            "query_type": state.get("query_type", "general"),
            "documents": [],
            "answer": "",
            "citations": [],
            "is_sufficient": False,
            "retry_queries": [],
            "retries": 0,
            "needs_vlm": False,
        }
        result = await sub_graph.ainvoke(sub_input)
        return _collect_sub_answer(result)

    async def summarize_node(state: AgentState) -> dict:
        return await summarize_conversation(state, llm=llm)

    async def classify_node(state: AgentState) -> dict:
        return await classify_query(state, llm=llm)

    async def analyze_node(state: AgentState) -> dict:
        return await analyze_query(state, llm=llm)

    graph = StateGraph(AgentState)

    async def synthesize_node(state: AgentState) -> dict:
        return await synthesize(state, llm=llm)

    graph.add_node("summarize", summarize_node)
    graph.add_node("classify", classify_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("sub_agent", sub_agent_node)
    graph.add_node("prepare_synthesis", prepare_synthesis)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "classify")
    graph.add_edge("classify", "analyze")
    graph.add_conditional_edges("analyze", dispatch, ["sub_agent"])
    graph.add_edge("sub_agent", "prepare_synthesis")
    graph.add_edge("prepare_synthesis", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)
