"""
聊天 API：基于 SSE 的流式多智能体问答。

事件类型（JSON 行，前端 api.js 解析）：
  session_id → status → sub_queries → answer（增量）→ citations → done | error
"""

import json
import uuid
import logging
from typing import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from config import Config
from app.dependencies import get_llm, get_retriever_tool, get_checkpointer
from app.store import create_session, get_session, update_session
from agent.graph import build_graph
from rag.citation import CitationExtractor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """POST /api/chat 请求体。"""
    query: str
    session_id: str | None = None  # 为空则服务端生成新 UUID


def _build_graph():
    """每次请求编译带 checkpointer 的 LangGraph（图结构相同，状态按 thread_id 隔离）。"""
    return build_graph(
        llm=get_llm(),
        retriever=get_retriever_tool(),
        citation_extractor=CitationExtractor,
        max_retries=Config.MAX_RETRIES,
        checkpointer=get_checkpointer(),
    )


async def _stream_response(graph, query: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    异步生成 SSE 数据行（每行一个 JSON 字符串）。

    stream_mode 同时使用 updates（节点输出）与 messages（LLM token 流），
    仅在 synthesize 节点累积并推送 answer 事件。
    """
    config = {"configurable": {"thread_id": session_id}}
    graph_input = {"query": query}

    yield json.dumps({"type": "session_id", "data": session_id})
    yield json.dumps({"type": "status", "data": "analyzing"})

    try:
        final_citations = []
        final_answer = ""
        answer_buf = ""

        async for chunk in graph.astream(graph_input, config=config, stream_mode=["updates", "messages"]):
            stream_type, data = chunk

            if stream_type == "updates":
                for node_name, node_output in data.items():
                    # 查询分解完成 → 通知前端展示子查询并切换状态
                    if node_name == "analyze":
                        sq = node_output.get("sub_queries", [])
                        if sq:
                            yield json.dumps({"type": "sub_queries", "data": sq})
                            yield json.dumps({"type": "status", "data": "searching"})

                    # 合成前已汇总全部引用
                    if node_name == "prepare_synthesis":
                        final_citations = node_output.get("citations", [])
                        logger.info(f"prepare_synthesis: {len(final_citations)} citations")

            elif stream_type == "messages":
                msg, metadata = data
                if metadata.get("langgraph_node") == "synthesize" and hasattr(msg, "content") and msg.content:
                    answer_buf += msg.content
                    yield json.dumps({"type": "answer", "data": answer_buf})

        final_answer = answer_buf

        if not final_answer:
            yield json.dumps({"type": "answer", "data": ""})

        # 流结束后写入 checkpoint：用户问 + 助手答（含 citations 元数据）
        await graph.aupdate_state(config, {
            "messages": [
                HumanMessage(content=query),
                AIMessage(content=final_answer, additional_kwargs={"citations": final_citations}),
            ],
            "answer": final_answer,
        })

        yield json.dumps({"type": "citations", "data": final_citations})

        # 首条消息时用问题前 50 字作为会话标题
        title_hint = query[:50] + ("…" if len(query) > 50 else "")
        session = await get_session(session_id)
        if session and not session.get("title"):
            await update_session(session_id, title=title_hint)

        yield json.dumps({"type": "done", "data": None})

    except Exception as e:
        logger.exception("Chat error")
        yield json.dumps({"type": "error", "data": str(e)})


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    流式对话入口。

    返回 EventSourceResponse，客户端需按 SSE 协议读取 data: 行。
    """
    session_id = req.session_id or str(uuid.uuid4())

    if not await get_session(session_id):
        await create_session(session_id)

    graph = _build_graph()

    return EventSourceResponse(
        _stream_response(graph, req.query, session_id),
        media_type="text/event-stream",
    )
