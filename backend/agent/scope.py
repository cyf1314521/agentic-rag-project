"""
方案 B — Scope 回复文案与 looks_like；在域判定委托 gateway（Corpus Gate）。
"""

from __future__ import annotations

import re
from typing import Literal

from .paper_scope import is_paper_id_only_query
from .text_utils import query_language

ScopeGate = Literal["continue", "out_of_scope"]

# 向后兼容（tri-state 已废弃）
ScopePre = Literal["in_scope", "out_of_scope", "ambiguous"]

_META_NOT_RAG = re.compile(
    r"有论文吗|有没有论文|有几篇|有哪些论文|上传了吗|"
    r"第[一二三四1234]篇|第一篇|第二篇|"
    r"有什么功能|你能做什么|怎么用|你是谁|你是什么|"
    r"^(你好|您好|hello|hi|谢谢|感谢)",
    re.I,
)

_PAPER_CONTENT_HINT = re.compile(
    r"摘要|abstract|方法|实验|结论|指标|章节|"
    r"说了什么|讲了什么|对比|比较|哪篇.*讲|"
    r"what does .+ say|compare|vs\.?",
    re.I,
)

_PAPER_ID_TOKEN = re.compile(r"eval_|paper", re.I)

# 仅 legacy 回退网关使用
_STANDALONE_OFF_DOMAIN_LEGACY = re.compile(
    r"比特币|btc|以太坊|加密货币|数字货币|股价|股票|大盘|"
    r"天气|温度|下雨|预报|"
    r"笑话|段子|写诗|对联|汇率|"
    r"写个|帮我写|生成|实现|debug|代码|python|javascript|脚本|程序|"
    r"bitcoin|ethereum|crypto|stock price|weather|"
    r"translate|翻译|help me code|write (a |an )?.+ script",
    re.I,
)

_ANAPHORIC_PREFIX = re.compile(
    r"^(那|那么|再|还|另外|继续|接着|刚才|上面|之前|此前|这个|那个|"
    r"这两点|第二|第一|同样|还有|"
    r"what about|how about|and the|more detail|elaborate|tell me more)\s*",
    re.I,
)
_FOLLOWUP_TAIL = re.compile(r"(呢|吗)\s*[？?]?$")


def is_meta_not_rag(query: str) -> bool:
    return bool(_META_NOT_RAG.search((query or "").strip()))


def _upload_hint(lang: str) -> str:
    if lang == "zh":
        return "请先上传 PDF 并绑定到当前会话，再提问论文内容。"
    return "Upload PDFs to this chat session before asking paper questions."


def scope_hint_message(session_paper_ids: list[str], query: str = "") -> str:
    lang = query_language(query)
    if not session_paper_ids:
        return _upload_hint(lang)
    papers = "\n".join(f"- `{pid}`" for pid in session_paper_ids)
    if lang == "zh":
        return (
            f"当前会话已绑定 {len(session_paper_ids)} 篇论文，例如：\n"
            f"「{session_paper_ids[0]} 的摘要说了什么？」\n\n"
            f"{papers}"
        )
    return (
        f"This session has {len(session_paper_ids)} paper(s). Example:\n"
        f"What does the abstract of {session_paper_ids[0]} say?\n\n"
        f"{papers}"
    )


def upload_required_reply(query: str = "") -> str:
    lang = query_language(query)
    if lang == "zh":
        return (
            "本系统是企业级论文 RAG，仅支持对已上传 PDF 的学术内容检索与问答。"
            + _upload_hint(lang)
        )
    return (
        "This is an enterprise paper RAG system for uploaded PDFs only. "
        + _upload_hint(lang)
    )


def out_of_scope_reply(query: str, session_paper_ids: list[str]) -> str:
    lang = query_language(query)
    if lang == "zh":
        base = (
            "本系统是企业级论文 RAG，仅支持对已上传 PDF 的学术内容检索与问答，"
            "不处理其他类型问题。请直接提问论文相关内容。"
        )
    else:
        base = (
            "This is an enterprise paper RAG system for uploaded PDFs only. "
            "Please ask questions about your papers."
        )
    if session_paper_ids:
        return f"{base}\n\n{scope_hint_message(session_paper_ids, query)}"
    return f"{base}\n\n{_upload_hint(lang)}"


def looks_like_paper_rag(query: str, session_paper_ids: list[str]) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if is_meta_not_rag(q):
        return False
    if re.search(r"哪篇", q):
        return True
    if _PAPER_CONTENT_HINT.search(q):
        return True
    if _PAPER_ID_TOKEN.search(q):
        return True
    if is_paper_id_only_query(q, session_paper_ids):
        return True
    q_lower = q.lower()
    for pid in session_paper_ids:
        if pid.lower() in q_lower:
            return True
    return False


def _ood_payload(query: str) -> str:
    q = (query or "").strip()
    q = _ANAPHORIC_PREFIX.sub("", q)
    q = _FOLLOWUP_TAIL.sub("", q).strip()
    return q


def is_standalone_off_domain(query: str) -> bool:
    """向后兼容：仅 legacy 网关使用。"""
    q = (query or "").strip()
    if not q:
        return False
    for candidate in (q, _ood_payload(q)):
        if candidate and _STANDALONE_OFF_DOMAIN_LEGACY.search(candidate):
            return True
    return False


def _legacy_scope_gate_action(query: str, session_paper_ids: list[str]) -> ScopeGate:
    q = (query or "").strip()
    if not q or not session_paper_ids:
        return "out_of_scope"
    if is_meta_not_rag(q):
        return "out_of_scope"
    if is_standalone_off_domain(q):
        return "out_of_scope"
    if looks_like_paper_rag(q, session_paper_ids):
        return "continue"
    from .query_rewrite import is_anaphoric_followup

    if is_anaphoric_followup(q):
        return "continue"
    return "out_of_scope"


def scope_gate(
    query: str,
    session_paper_ids: list[str],
    *,
    messages: list | None = None,
    pending_user_query: str = "",
    summary: str = "",
    embeddings: object | None = None,
    corpus: object | None = None,
) -> ScopeGate:
    """委托 gateway.run_gateway；upload_required 映射为 out_of_scope（由 CheckSession 处理）。"""
    from .gateway import run_gateway

    decision = run_gateway(
        query,
        session_paper_ids,
        messages=messages,
        pending_user_query=pending_user_query,
        summary=summary,
        embeddings=embeddings,
        corpus=corpus,
    )
    if decision.action == "continue":
        return "continue"
    return "out_of_scope"


def scope_check_pre_llm(query: str, session_paper_ids: list[str]) -> ScopePre:
    if scope_gate(query, session_paper_ids) == "continue":
        return "in_scope"
    return "out_of_scope"
