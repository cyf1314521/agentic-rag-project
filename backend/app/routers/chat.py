"""
聊天 API：基于 SSE 的流式多智能体问答。

事件类型（JSON 行，前端 api.js 解析）：
  session_id → turn_id → status → clarification? → sub_queries → sub_agent_status →
  warnings → answer（增量）→ citations → done | error
"""

import json
import uuid
import logging
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from config import Config
from app.dependencies import get_llm, get_retriever_tool, get_checkpointer
from app.store import create_session, get_session, update_session, get_session_paper_ids
from agent.graph import build_graph
from agent.nodes import sanitize_query
from agent.states import fresh_turn_state
from agent.resilience import is_failed_sub_answer, set_request_deadline, clear_request_deadline
from rag.citation import CitationExtractor, sanitize_answer_citations
from rag.chat_trace import start_trace, finish_trace, get_trace

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
    trace_id = uuid.uuid4().hex[:12]
    query = sanitize_query(query)
    paper_ids = await get_session_paper_ids(session_id)
    if paper_ids:
        logger.info("Session scope: %d paper(s) %s", len(paper_ids), paper_ids)
    else:
        logger.info(
            "Session %s has no linked papers — only non-RAG replies until upload.",
            session_id,
        )
    turn_id = uuid.uuid4().hex[:16]
    graph_input = fresh_turn_state(query, trace_id, paper_ids=paper_ids, turn_id=turn_id)
    start_trace(session_id, query, trace_id=trace_id)
    set_request_deadline(float(Config.REQUEST_TIMEOUT))

    yield json.dumps({"type": "session_id", "data": session_id})
    yield json.dumps({"type": "turn_id", "data": turn_id})
    yield json.dumps({"type": "status", "data": "analyzing"})

    final_citations: list = []
    final_answer = ""
    answer_buf = ""
    failed_sub_queries: list[dict] = []
    partial_failure = False
    needs_clarification = False
    direct_response = False
    checkpoint_saved = False

    async def _persist_turn(answer_text: str) -> None:
        nonlocal checkpoint_saved
        if checkpoint_saved:
            return
        await graph.aupdate_state(config, {
            "messages": [
                HumanMessage(content=query),
                AIMessage(
                    content=answer_text,
                    additional_kwargs={
                        "citations": final_citations,
                        "turn_id": turn_id,
                        "partial_failure": partial_failure,
                        "failed_sub_queries": failed_sub_queries,
                    },
                ),
            ],
            "answer": answer_text,
            "turn_id": turn_id,
            "failed_sub_queries": failed_sub_queries,
        })
        checkpoint_saved = True

    try:
        async for chunk in graph.astream(graph_input, config=config, stream_mode=["updates", "messages"]):
            stream_type, data = chunk

            if stream_type == "updates":
                for node_name, node_output in data.items():
                    if node_output is None:
                        continue
                    gateway_fields = {}
                    gw = node_output.get("gateway") or {}
                    if gw.get("content_score") is not None:
                        gateway_fields["content_score"] = gw["content_score"]
                    if gw.get("reason"):
                        gateway_fields["gateway_reason"] = gw["reason"]

                    if node_name == "compliance_gate" and node_output.get("compliance_blocked"):
                        direct_response = True
                        yield json.dumps({
                            "type": "intent",
                            "data": {
                                "intent": node_output.get("intent", "out_of_domain"),
                                "focus_paper_ids": [],
                                "retrieval_mode": "none",
                                "direct_response": True,
                                "compliance_blocked": True,
                                "turn_state": node_output.get("turn_state", "out_of_scope"),
                            },
                        })
                        yield json.dumps({"type": "status", "data": "direct"})

                    if node_name == "resolve_intent":
                        if node_output.get("direct_response"):
                            direct_response = True
                            yield json.dumps({
                                "type": "intent",
                                "data": {
                                    "intent": node_output.get("intent"),
                                    "focus_paper_ids": [],
                                    "retrieval_mode": "none",
                                    "direct_response": True,
                                    "turn_state": node_output.get("turn_state", "out_of_scope"),
                                    **gateway_fields,
                                },
                            })
                            yield json.dumps({"type": "status", "data": "direct"})
                        elif node_output.get("needs_clarification"):
                            needs_clarification = True
                            yield json.dumps({
                                "type": "clarification",
                                "data": {
                                    "paper_ids": paper_ids,
                                    "candidates": node_output.get("candidate_paper_ids") or paper_ids,
                                    "scope_mode": node_output.get("scope_mode", "ambiguous"),
                                    "intent": node_output.get("intent", ""),
                                },
                            })
                            yield json.dumps({"type": "status", "data": "clarification"})
                        elif node_output.get("intent"):
                            yield json.dumps({
                                "type": "intent",
                                "data": {
                                    "intent": node_output.get("intent"),
                                    "focus_paper_ids": node_output.get("focus_paper_ids") or [],
                                    "retrieval_mode": node_output.get("retrieval_mode", "body"),
                                    "turn_state": node_output.get("turn_state", "rag_ready"),
                                    **gateway_fields,
                                },
                            })

                    if node_name in ("clarify", "respond_direct"):
                        direct_msg = node_output.get("answer", "")
                        if direct_msg:
                            answer_buf = direct_msg
                            yield json.dumps({"type": "answer", "data": direct_msg})

                    # 查询分解完成 → 通知前端展示子查询并切换状态
                    if node_name == "analyze":
                        sq = node_output.get("sub_queries", [])
                        if sq:
                            yield json.dumps({"type": "sub_queries", "data": sq})
                            yield json.dumps({"type": "status", "data": "searching"})

                    # 每路子 Agent 完成（成功或 Fallback 失败占位）
                    if node_name == "sub_agent":
                        for sa in node_output.get("sub_answers", []):
                            status = "failed" if is_failed_sub_answer(sa) else "ok"
                            if status == "failed":
                                partial_failure = True
                            yield json.dumps({
                                "type": "sub_agent_status",
                                "data": {
                                    "query": sa.get("query", ""),
                                    "status": status,
                                    "error": sa.get("error", ""),
                                },
                            })

                    # 合成前已汇总全部引用
                    if node_name == "prepare_synthesis":
                        final_citations = node_output.get("citations", [])
                        failed_sub_queries = node_output.get("failed_sub_queries", [])
                        if failed_sub_queries:
                            partial_failure = True
                            yield json.dumps({
                                "type": "warnings",
                                "data": [
                                    f"Sub-query retrieval failed: {item.get('query', '')} ({item.get('error', '')})"
                                    for item in failed_sub_queries
                                ],
                            })
                        logger.info(
                            "prepare_synthesis: %d citations, %d failed sub-queries",
                            len(final_citations),
                            len(failed_sub_queries),
                        )

                    # synthesize 走子答案直出时不会触发 LLM token 流，需从节点输出取 answer
                    if node_name == "synthesize":
                        synth_answer = node_output.get("answer", "") or ""
                        if synth_answer:
                            answer_buf = synth_answer
                            yield json.dumps({"type": "answer", "data": answer_buf})

            elif stream_type == "messages":
                msg, metadata = data
                if metadata.get("langgraph_node") == "synthesize" and hasattr(msg, "content") and msg.content:
                    answer_buf += msg.content
                    yield json.dumps({"type": "answer", "data": answer_buf})

        final_answer = answer_buf

        if final_answer and final_citations:
            final_answer, stripped = sanitize_answer_citations(
                final_answer, len(final_citations),
            )
            if stripped:
                logger.warning(
                    "Chat stream stripped invalid citation indices: %s",
                    stripped,
                )
                yield json.dumps({"type": "answer", "data": final_answer})

        if not final_answer:
            yield json.dumps({"type": "answer", "data": ""})

        await _persist_turn(final_answer)
        checkpoint_saved = True

        yield json.dumps({"type": "citations", "data": final_citations})

        # 首条消息时用问题前 50 字作为会话标题
        title_hint = query[:50] + ("…" if len(query) > 50 else "")
        session = await get_session(session_id)
        if session and not session.get("title"):
            await update_session(session_id, title=title_hint)

        yield json.dumps({
            "type": "done",
            "data": {
                "partial": partial_failure,
                "failed_count": len(failed_sub_queries),
                "needs_clarification": needs_clarification,
                "direct_response": direct_response,
            },
        })

    except asyncio.CancelledError:
        logger.warning("Chat stream cancelled (client disconnect) session=%s", session_id)
        if answer_buf.strip() and not checkpoint_saved:
            try:
                clean = answer_buf
                if final_citations:
                    clean, _ = sanitize_answer_citations(clean, len(final_citations))
                await _persist_turn(clean)
            except Exception:
                logger.exception("Failed to persist partial answer on disconnect")
        raise
    except Exception as e:
        logger.exception("Chat error")
        tr = get_trace(trace_id)
        if tr:
            tr.error(str(e))
        yield json.dumps({"type": "error", "data": str(e)})
    finally:
        clear_request_deadline()
        finish_trace(trace_id)


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
