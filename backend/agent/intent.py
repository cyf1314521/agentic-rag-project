"""
LLM 检索意图解析：基于会话上下文输出槽位；语义只在此处判断，不做规则猜主题。
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import Config
from .prompts import INTENT_RESOLVER
from .query_rewrite import pairs_from_messages, resolve_effective_query
from .paper_scope import is_paper_id_only_query
from .states import AgentState
from .text_utils import query_language

logger = logging.getLogger(__name__)

_DISCOVERY_HINTS = re.compile(
    r"哪篇|哪些篇|有没有.+论文|是否存在|哪篇文章|"
    r"which paper|what paper|find (the |a )?paper|papers (about|on|that)",
    re.I,
)
_COMPARE_HINTS = re.compile(
    r"对比|比较|区别|差异|versus|vs\.?|compare",
    re.I,
)

IntentKind = Literal[
    "paper_qa",
    "paper_discovery",
    "paper_compare",
    "need_clarification",
]

MissingSlot = Literal[
    "which_paper",
    "what_to_ask",
    "intent_unclear",
]


class RetrievalIntent(BaseModel):
    """resolve_intent LLM 结构化输出（槽位 + 置信度）。"""

    effective_query: str = Field(
        description="Standalone retrieval query after merging conversation context",
    )
    intent: IntentKind = Field(
        description="paper_qa | paper_discovery | paper_compare | need_clarification",
    )
    focus_paper_ids: list[str] = Field(
        default_factory=list,
        description="Target paper ids from context; empty if not yet identifiable",
    )
    missing: list[str] = Field(
        default_factory=list,
        description='Empty if ready to retrieve; else slots like "which_paper", "what_to_ask"',
    )
    clarification_question: str = Field(
        default="",
        description="User-facing question when missing is non-empty; same language as user",
    )
    constraints: dict[str, str] = Field(
        default_factory=dict,
        description='Optional filters e.g. {"time": "latest", "version": "3.3", "topic": "quantum"}',
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


def _recent_dialogue(messages: list, limit: int = 8) -> str:
    pairs = pairs_from_messages(messages)[-limit:]
    lines: list[str] = []
    for role, content in pairs:
        label = "User" if role == "human" else "Assistant"
        text = (content or "").strip()
        if text:
            lines.append(f"{label}: {text[:500]}")
    return "\n".join(lines) or "(no prior turns)"


def _fast_path_intent(
    query: str,
    session_paper_ids: list[str],
) -> RetrievalIntent | None:
    """单篇会话：scope 由产品固定，仅需 LLM 处理复杂多轮；简单完整问句可走快路径。"""
    if len(session_paper_ids) != 1:
        return None
    q = (query or "").strip()
    if not q:
        return None
    # 仅 paper_id、无具体问题 → 交给 LLM 填 missing=what_to_ask
    if is_paper_id_only_query(q, session_paper_ids):
        return None
    if _DISCOVERY_HINTS.search(q) or _COMPARE_HINTS.search(q):
        return None
    return RetrievalIntent(
        effective_query=q,
        intent="paper_qa",
        focus_paper_ids=list(session_paper_ids),
        missing=[],
        confidence=0.95,
    )


def _fallback_clarify_message(query: str) -> str:
    if query_language(query) == "zh":
        return "我无法从当前对话确定您的具体意图，请补充说明要查询的论文或问题。"
    return (
        "I cannot determine your intent from the conversation. "
        "Please specify which paper and what you want to know."
    )


async def resolve_intent_llm(
    state: AgentState,
    llm: BaseChatModel,
    *,
    effective_query: str,
    include_pending: bool = True,
) -> RetrievalIntent:
    """调用 LLM，基于全上下文解析意图槽位。"""
    session_paper_ids = state.get("paper_ids") or []
    fast = _fast_path_intent(effective_query, session_paper_ids)
    if fast is not None:
        return fast

    papers = ", ".join(session_paper_ids) if session_paper_ids else "(none)"
    pending = (state.get("pending_user_query") or "").strip()
    user_block = (
        f"Session papers (only these ids are valid): {papers}\n"
        f"Conversation summary:\n{(state.get('summary') or '(none)')[:600]}\n\n"
        f"Recent dialogue:\n{_recent_dialogue(state.get('messages', []))}\n"
    )
    if pending and include_pending:
        user_block += f"\nPending question before last clarification: {pending}\n"
    user_block += f"\nCurrent user message: {effective_query}"

    structured = llm.with_structured_output(RetrievalIntent)
    try:
        result = await structured.ainvoke([
            SystemMessage(content=INTENT_RESOLVER),
            HumanMessage(content=user_block),
        ])
        intent = RetrievalIntent.model_validate(result)
        if not intent.effective_query.strip():
            intent.effective_query = effective_query
        return intent
    except Exception as exc:
        logger.warning("LLM intent resolution failed: %s", exc)
        return RetrievalIntent(
            effective_query=effective_query,
            intent="need_clarification",
            focus_paper_ids=[],
            missing=["intent_unclear"],
            clarification_question=_fallback_clarify_message(effective_query),
            confidence=0.0,
        )


def rewrite_query_for_intent(state: AgentState) -> tuple[str, str, str]:
    """
    方案 B Phase 2：仅做多轮 query 合并，不做路由。

    返回 (raw_query, effective_query, rewrite_reason)。
    """
    session_paper_ids = state.get("paper_ids") or []
    raw_query = (state.get("query") or "").strip()
    effective_query, rewrite_reason = resolve_effective_query(
        raw_query,
        session_paper_ids,
        pairs_from_messages(state.get("messages", [])),
        summary=state.get("summary", ""),
        pending_user_query=state.get("pending_user_query", ""),
    )
    return raw_query, effective_query, rewrite_reason


async def build_retrieval_intent(
    state: AgentState,
    llm: BaseChatModel,
) -> tuple[RetrievalIntent, str, str]:
    """
    兼容入口：rewrite + LLM intent（路由由 turn_resolver 负责）。
    """
    raw_query, effective_query, rewrite_reason = rewrite_query_for_intent(state)
    include_pending = rewrite_reason != "topic_switch"
    intent = await resolve_intent_llm(
        state, llm, effective_query=effective_query, include_pending=include_pending,
    )
    if intent.effective_query.strip() != effective_query:
        effective_query = intent.effective_query.strip() or effective_query
        intent.effective_query = effective_query
    return intent, raw_query, rewrite_reason
