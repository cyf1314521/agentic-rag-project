"""
检索前 query 改写：把多轮对话压缩成单轮可检索的完整问题。

在 resolve_intent / retrieve 之前运行，写回 state["query"]。
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage

from .paper_scope import is_paper_id_only_query
from .scope import is_meta_not_rag

RewriteReason = Literal[
    "",
    "clarification_followup",
    "anaphoric_followup",
    "clarification+anaphoric",
    "topic_switch",
]

_CLARIFICATION_HINTS = re.compile(
    r"无法确定您指的是哪一篇|请指明要查询的论文|本会话可用论文|"
    r"specify which one|please specify",
    re.I,
)
_DEICTIC_PAPER = re.compile(
    r"这篇论文|该论文|该篇|此文|本文|this paper|the paper",
    re.I,
)
_ANAPHORIC_FOLLOWUP = re.compile(
    r"^(那|那么|再|还|另外|继续|接着|刚才|上面|之前|此前|这个|那个|"
    r"这两点|第二|第一|同样|还有|"
    r"what about|how about|and the|more detail|elaborate|tell me more)",
    re.I,
)
_FOLLOWUP_TAIL = re.compile(r"(呢|吗)\s*[？?]?$")
_TOPIC_SWITCH_DISMISS = re.compile(
    r"^(算了|不用了|别管了|换个话题|不想看了|forget it|never mind|stop)\b",
    re.I,
)
_INDEPENDENT_TASK = re.compile(
    r"(写个|帮我写|生成|实现|debug|代码|python|javascript|脚本|程序|"
    r"translate|翻译|写诗|笑话|天气|"
    r"write (a |an )?.+ script|implement .+ in|help me code)",
    re.I,
)
_GREETING_ONLY = re.compile(r"^(你好|您好|hello|hi|谢谢|感谢)", re.I)
_PAPER_RAG_HINTS = re.compile(
    r"论文|摘要|abstract|paper|eval_|章节|方法|实验|结论|"
    r"quantum|blockchain|wind|battery|protein",
    re.I,
)


def pairs_from_messages(messages: list) -> list[tuple[str, str]]:
    """LangChain messages → [("human"|"ai", content), ...]。"""
    pairs: list[tuple[str, str]] = []
    for msg in messages or []:
        if isinstance(msg, HumanMessage):
            pairs.append(("human", str(msg.content)))
        elif isinstance(msg, AIMessage):
            pairs.append(("ai", str(msg.content)))
    return pairs


def _last_substantive_human(
    recent_messages: list[tuple[str, str]],
    session_paper_ids: list[str],
) -> str | None:
    """跳过澄清话术与纯 paper_id 回复，取最近一条真实用户问题。"""
    for role, content in reversed(recent_messages):
        if role != "human":
            continue
        text = (content or "").strip()
        if not text or is_paper_id_only_query(text, session_paper_ids):
            continue
        if is_meta_not_rag(text):
            continue
        return text
    return None


def is_topic_switch(
    query: str,
    *,
    pending_user_query: str = "",
    session_paper_ids: list[str] | None = None,
) -> bool:
    """
    检测用户是否放弃 pending 澄清、转向独立新任务（代码/闲聊等）。
    有 pending 时更敏感；无 pending 时仅拦截明显离题指令。
    """
    q = (query or "").strip()
    if not q:
        return False

    pending = (pending_user_query or "").strip()
    if pending and _TOPIC_SWITCH_DISMISS.search(q):
        return True

    if _INDEPENDENT_TASK.search(q) and not _PAPER_RAG_HINTS.search(q):
        return True

    if pending and len(q) >= 18 and not _PAPER_RAG_HINTS.search(q):
        if is_paper_id_only_query(q, session_paper_ids or []):
            return False
        if not _ANAPHORIC_FOLLOWUP.search(q) and not _DEICTIC_PAPER.search(q):
            return True

    return False


def is_anaphoric_followup(query: str) -> bool:
    """短句、指代词开头、或句末「呢/吗」等，视为依赖上一轮的追问。"""
    q = (query or "").strip()
    if not q or len(q) > 56:
        return False
    if is_topic_switch(q):
        return False
    if _GREETING_ONLY.match(q):
        return False
    if _TOPIC_SWITCH_DISMISS.search(q):
        return False
    if _INDEPENDENT_TASK.search(q):
        return False
    if _ANAPHORIC_FOLLOWUP.search(q):
        return True
    if _FOLLOWUP_TAIL.search(q) and len(q) <= 28:
        return True
    if len(q) <= 12 and not re.search(r"eval_|摘要|abstract", q, re.I):
        if _GREETING_ONLY.match(q):
            return False
        return True
    return False


def merge_clarification_followup(
    query: str,
    session_paper_ids: list[str],
    recent_messages: list[tuple[str, str]],
    *,
    pending_user_query: str = "",
) -> str:
    """澄清后用户只回复 paper_id 时，与上一轮真实问题合并。"""
    pid = is_paper_id_only_query(query, session_paper_ids)
    if not pid:
        return query

    prev_human: str | None = None
    saw_clarification = False
    for role, content in reversed(recent_messages):
        if role == "ai" and _CLARIFICATION_HINTS.search(content):
            saw_clarification = True
            continue
        if role != "human":
            continue
        if is_paper_id_only_query(content, session_paper_ids):
            continue
        if saw_clarification or prev_human is None:
            prev_human = content
            if saw_clarification:
                break

    if not prev_human and pending_user_query.strip():
        prev_human = pending_user_query.strip()

    if not prev_human:
        return query

    merged = _DEICTIC_PAPER.sub(pid, prev_human)
    if pid.lower() not in merged.lower():
        merged = f"{pid} {merged}"
    return merged.strip()


def merge_anaphoric_followup(
    query: str,
    session_paper_ids: list[str],
    recent_messages: list[tuple[str, str]],
    *,
    summary: str = "",
) -> str:
    """「那 P99 呢？」类追问：与上一轮用户问题合并为可检索的完整句。"""
    q = (query or "").strip()
    if not is_anaphoric_followup(q):
        return query

    prev = _last_substantive_human(recent_messages, session_paper_ids)
    if not prev:
        if summary:
            snippet = summary.strip()[:400]
            return f"（对话背景：{snippet}）{q}"
        return query

    if q in prev or prev in q:
        return query

    expanded = _DEICTIC_PAPER.sub(prev, q) if _DEICTIC_PAPER.search(q) else q
    if expanded != q and expanded.lower() != prev.lower():
        return expanded.strip()

    return f"{prev}；追问：{q}"


def resolve_effective_query(
    query: str,
    session_paper_ids: list[str],
    recent_messages: list[tuple[str, str]],
    *,
    summary: str = "",
    pending_user_query: str = "",
) -> tuple[str, RewriteReason]:
    """
    检索前统一入口：意图切换检测 → 澄清补全 → 指代追问合并。

    返回 (effective_query, rewrite_reason)。
    """
    original = (query or "").strip()
    reason: RewriteReason = ""

    if is_topic_switch(
        original,
        pending_user_query=pending_user_query,
        session_paper_ids=session_paper_ids,
    ):
        return original, "topic_switch"

    q = merge_clarification_followup(
        original, session_paper_ids, recent_messages,
        pending_user_query=pending_user_query,
    )
    if q != original:
        reason = "clarification_followup"

    q2 = merge_anaphoric_followup(
        q, session_paper_ids, recent_messages, summary=summary,
    )
    if q2 != q:
        if reason == "clarification_followup":
            reason = "clarification+anaphoric"
        else:
            reason = "anaphoric_followup"
        q = q2

    return q, reason
