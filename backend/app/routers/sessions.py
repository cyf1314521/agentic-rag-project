"""
会话管理 API。

会话列表与标题存在 PostgreSQL sessions 表；
完整多轮消息从 LangGraph checkpointer 按 thread_id（= session_id）恢复。
"""

import logging
from typing import cast

from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.store import (
    list_sessions,
    get_session,
    delete_session,
    get_session_paper_ids,
    create_session,
    link_session_paper,
)
from app.dependencies import get_checkpointer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class LinkPaperRequest(BaseModel):
    """POST /api/sessions/{id}/papers — 将已入库 paper 绑定到会话 scope（评测/联调）。"""
    paper_id: str = Field(..., min_length=1)


@router.get("")
async def get_sessions():
    """GET /api/sessions — 列出所有会话（按 updated_at 降序）。"""
    return await list_sessions()


@router.get("/{session_id}")
async def get_session_detail(session_id: str):
    """GET /api/sessions/{id} — 会话元数据（不含消息正文）。"""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}/papers")
async def link_paper_to_session(session_id: str, body: LinkPaperRequest):
    """
    绑定 paper_id 到会话（与上传 PDF 后的 link_session_paper 相同）。

    用于评测脚本：Milvus 中已有 eval_* 文档时，无需重复上传即可启用 session scope。
    """
    if not await get_session(session_id):
        await create_session(session_id)
    await link_session_paper(session_id, body.paper_id)
    paper_ids = await get_session_paper_ids(session_id)
    return {"session_id": session_id, "paper_ids": paper_ids}


@router.get("/{session_id}/papers")
async def get_session_papers(session_id: str):
    """GET /api/sessions/{id}/papers — 本会话绑定的 paper_id（检索 scope）。"""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    paper_ids = await get_session_paper_ids(session_id)
    return {"session_id": session_id, "paper_ids": paper_ids}


@router.get("/{session_id}/history")
async def get_history(session_id: str):
    """
    GET /api/sessions/{id}/history — 从 checkpoint 重建对话历史。

    返回格式：{ session_id, messages: [{ role, content, citations? }, ...] }
    """
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    config = cast(RunnableConfig, {"configurable": {"thread_id": session_id}})
    checkpointer = get_checkpointer()
    try:
        checkpoint = await checkpointer.aget(config)
    except Exception:
        checkpoint = None

    if not checkpoint or not checkpoint.get("channel_values"):
        return {"session_id": session_id, "messages": []}

    messages = []
    for msg in checkpoint["channel_values"].get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        else:
            # AIMessage：引用存在 additional_kwargs
            citations = msg.additional_kwargs.get("citations", [])
            messages.append({"role": "assistant", "content": msg.content, "citations": citations})

    return {"session_id": session_id, "messages": messages}


@router.delete("/{session_id}")
async def remove_session(session_id: str):
    """DELETE /api/sessions/{id} — 删除元数据并清理 LangGraph thread checkpoint。"""
    ok = await delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        await get_checkpointer().adelete_thread(session_id)
    except Exception as e:
        logger.warning(f"Failed to clean checkpoint for {session_id}: {e}")

    return {"ok": True}
