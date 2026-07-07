"""
LangGraph 节点函数实现。

分两类：
1. 主图节点：summarize / resolve_intent / analyze / prepare_synthesis / synthesize
2. 子图节点：retrieve / generate / reflect / prepare_retry / should_retry

依赖 Pydantic 结构化输出（QueryAnalysis、ReflectionResult）
保证 LLM 返回可解析 JSON。
"""

import re
import logging
import asyncio
from typing import Any, cast

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document

from .states import AgentState, SubAgentState
from .resilience import (
    is_failed_sub_answer,
    retry_async,
    retry_sync,
    effective_timeout,
    parse_backoff_ms,
)
from config import Config
from app.llm_utils import message_content_to_str
from .paper_scope import (
    boost_docs_for_focus_paper,
    effective_retrieval_paper_ids,
)
from .intent import RetrievalIntent, rewrite_query_for_intent, resolve_intent_llm
from .turn_resolver import resolve_turn
from .gateway import run_gateway
from .text_utils import COMPLEX_HINTS, query_language
from .prompts import QUERY_ANALYZER, SYNTHESIZER, GENERATOR, REFLECTOR, SUMMARIZER
from rag.chat_trace import get_trace
from rag.citation import (
    CitationExtractor,
    is_citation_only_answer,
    parse_citation_indices,
    sanitize_answer_citations,
)
from rag.evidence_gate import evaluate_evidence
from rag.grounding_check import check_grounding
from .compliance_gate import check_compliance, compliance_blocked_reply

logger = logging.getLogger(__name__)

# 送入本地 LLM 的检索上下文上限，避免 Ollama 因 prompt 过长返回 502
MAX_GENERATE_CONTEXT_CHARS = 12_000

# 保留最近 N 条消息原文；更早的由 summarize 压缩进 summary
WINDOW_SIZE = 6

_INSUFFICIENT_PHRASES = (
    "does not contain sufficient information",
    "insufficient information",
    "信息不足",
    "没有包含足够",
    "没有足够的信息",
    "没有足够的",
    "提供的上下文没有",
    "上下文没有包含",
    "无法回答",
    "cannot answer",
    "no relevant information",
    "no relevant information found",
)

_SIMPLE_HINTS = re.compile(
    r"摘要|abstract|关键词|keyword|说了什么|是什么|多少|哪篇|第几页|"
    r"what does .+ say|what is the abstract",
    re.I,
)
_ABSTRACT_HINTS = re.compile(r"摘要|abstract", re.I)
_SINGLE_PAPER_QUERY_HINTS = re.compile(
    r"这篇|该论文|该篇|此文|本文|this paper|the paper|摘要里|摘要中|摘要采用|摘要提到|摘要表示|"
    r"abstract says|in the abstract",
    re.I,
)

MAX_SUB_QUERIES_SIMPLE = 1
MAX_SUB_QUERIES_COMPLEX = 2

_RETRIEVE_BACKOFF_MS = parse_backoff_ms(Config.RETRIEVE_RETRY_BACKOFF_MS, (200, 500, 1000))
_LLM_NODE_BACKOFF_MS = parse_backoff_ms(Config.LLM_NODE_RETRY_BACKOFF_MS, (1000, 3000))

# 并行子 Agent 共享 GPU embedding 时串行检索，避免多路同时 embed 导致步超时
_RETRIEVE_SEM = asyncio.Semaphore(1)

_INSUFFICIENT_TAIL_RE = re.compile(
    r"\s*(?:The provided context does not contain sufficient information.*?|"
    r"This question cannot be answered.*?|"
    r"I cannot answer.*?)\s*$",
    re.I | re.DOTALL,
)
_PROMPT_LEAK_RE = re.compile(
    r"\bDualPath\b|KV cache splitting|standard attention",
    re.I,
)
_SOURCE_LINE_RE = re.compile(r"\[Source:[^\]]*\]", re.I)


async def _invoke_llm_with_retry(
    llm: Any,
    messages: list,
    *,
    operation: str,
    sub_query: str,
    trace_id: str,
    step_timeout: float,
):
    """子图 generate / reflect 的 LLM 调用：步超时 + 有界重试。"""
    trace = get_trace(trace_id)

    async def _once():
        return await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=effective_timeout(step_timeout),
        )

    def _on_retry(attempt: int, exc: BaseException) -> None:
        if trace:
            trace.retry_attempt(
                operation=operation,
                attempt=attempt,
                max_attempts=Config.LLM_NODE_MAX_RETRIES,
                error=f"{type(exc).__name__}: {exc}",
                sub_query=sub_query,
            )

    return await retry_async(
        _once,
        label=f"{operation}({sub_query[:30]!r})",
        max_attempts=Config.LLM_NODE_MAX_RETRIES,
        backoff_ms=_LLM_NODE_BACKOFF_MS,
        on_retry=_on_retry,
    )


def _is_insufficient_answer(text: str) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in _INSUFFICIENT_PHRASES)


def _strip_insufficient_boilerplate(text: str) -> str:
    """裁掉 generate 尾部英文「信息不足」模板句，保留已生成的有效内容。"""
    cleaned = _INSUFFICIENT_TAIL_RE.sub("", text or "").strip()
    cleaned = re.sub(r"\[\d+\]\s*未提供相关信息\.?", "", cleaned).strip()
    return cleaned or (text or "").strip()


def _sanitize_generate_answer(answer: str, context: str) -> str:
    """去掉 prompt 示例泄漏、Source 行抄录、[i] 占位符等脏输出。"""
    text = _strip_insufficient_boilerplate(answer)
    text = _SOURCE_LINE_RE.sub("", text)
    text = re.sub(r"\[i\]", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    if _PROMPT_LEAK_RE.search(text) and not _PROMPT_LEAK_RE.search(context):
        # 保留含 [n] 引用且像正常答案的句子，丢弃 DualPath 等幻觉句
        kept: list[str] = []
        for part in re.split(r"(?<=[。！？.!?])\s+", text):
            part = part.strip()
            if not part:
                continue
            if _PROMPT_LEAK_RE.search(part) and not re.search(r"\[\d+\]", part):
                continue
            kept.append(part)
        text = " ".join(kept).strip()

    return text or _strip_insufficient_boilerplate(answer)


def _ensure_primary_citation(answer: str, documents: list[str], query: str) -> str:
    """模型未写 [n] 时，为实质性答案补上最相关段落编号。"""
    text = (answer or "").strip()
    if not text or not documents or parse_citation_indices(text):
        return answer
    if len(text) < 20:
        return answer

    keywords: list[str] = []
    q = query or ""
    if "摘要" in q or "abstract" in q.lower():
        keywords.extend(["摘要", "Abstract", "abstract"])

    best_idx = 1
    best_score = -1
    prefix = text[:100]
    for i, doc in enumerate(documents, 1):
        score = sum(1 for kw in keywords if kw in doc)
        if prefix and prefix in doc:
            score += 5
        if score > best_score:
            best_score = score
            best_idx = i

    return text.rstrip() + f" [{best_idx}]"


def sanitize_query(query: str) -> str:
    """去掉前端/JSON 误入的尾部引号与反斜杠。"""
    q = (query or "").strip()
    q = re.sub(r'\\+"\s*$', "", q)
    q = re.sub(r'"\s*$', "", q)
    return q.strip()


def _heuristic_complexity(query: str) -> str | None:
    if COMPLEX_HINTS.search(query):
        return "complex"
    if _SIMPLE_HINTS.search(query):
        return "simple"
    return None


def _normalize_sub_queries(query: str, sub_queries: list[str], complexity: str) -> list[str]:
    lang = query_language(query)
    max_n = MAX_SUB_QUERIES_SIMPLE if complexity == "simple" else MAX_SUB_QUERIES_COMPLEX
    out: list[str] = []
    seen: set[str] = set()
    for sq in sub_queries:
        sq = sanitize_query(sq or "")
        if not sq or sq in seen:
            continue
        if lang == "zh" and not re.search(r"[\u4e00-\u9fff]", sq):
            continue
        seen.add(sq)
        out.append(sq)
    if query not in out:
        out.insert(0, query)
    out = out[:max_n]
    if complexity == "simple":
        return [query]
    return out[:MAX_SUB_QUERIES_COMPLEX]


def _retrieval_query_variants(
    query: str,
    queries: list[str],
    *,
    focus_paper_ids: list[str] | None = None,
) -> list[str]:
    """摘要类问题追加检索词，提升命中 摘要/Abstract 段落。"""
    if focus_paper_ids:
        return queries[:1]
    if len(query) >= 28:
        return queries[:1]
    if not _ABSTRACT_HINTS.search(query):
        return queries
    extra = []
    if query_language(query) == "zh":
        extra = ["摘要", "abstract 摘要"]
    else:
        extra = ["abstract", "summary abstract"]
    out = list(queries)
    for e in extra:
        if e not in out:
            out.append(e)
    return out[:4]


def _sub_answers_for_turn(state: AgentState) -> list:
    allowed = set(state.get("sub_queries") or [])
    if not allowed:
        return list(state.get("sub_answers", []))
    return [sa for sa in state.get("sub_answers", []) if sa.get("query") in allowed]


def _dedupe_citations(citations: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in citations:
        key = str(c.get("chunk_id") or id(c))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _collect_valid_evidence(sub_answers: list) -> list:
    valid = []
    for sa in sub_answers:
        if is_failed_sub_answer(sa):
            continue
        ans = str(sa.get("answer", "")).strip()
        if len(ans) < 30 or is_citation_only_answer(ans):
            continue
        if ans.lower() == "no relevant information found.":
            continue
        if _is_insufficient_answer(ans) and not (
            re.search(r"\[\d+\]", ans) and len(ans) >= 40
        ):
            continue
        valid.append(sa)
    return valid


def _rule_reflect_sufficient(answer: str) -> bool:
    if _is_insufficient_answer(answer):
        return False
    if re.search(r"\[\d+\]", answer) and len(answer.strip()) >= 40:
        return True
    return len(answer.strip()) >= 80


class QueryAnalysis(BaseModel):
    """analyze_query 的结构化输出。"""
    complexity: str = Field(description="simple or complex")
    sub_queries: list[str] = Field(description="Sub-queries for retrieval, original first")


class ReflectionResult(BaseModel):
    """reflect 节点的结构化输出。"""
    is_sufficient: bool = Field(description="Whether the answer is sufficient")
    retry_queries: list[str] = Field(default_factory=list, description="Queries for missing info")


# --------------- 主图节点 ---------------


async def compliance_gate_node(state: AgentState) -> dict:
    """可选合规前置：越狱/敏感词拦截，默认关闭。"""
    result = check_compliance(state.get("query", ""))
    if not result.blocked:
        return {"compliance_blocked": False}

    msg = compliance_blocked_reply(state.get("query", ""))
    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.compliance_block(reason=result.reason)
    return {
        "compliance_blocked": True,
        "direct_response": True,
        "turn_state": "out_of_scope",
        "answer": msg,
        "intent": "out_of_domain",
        "retrieval_mode": "none",
    }


def route_after_compliance(state: AgentState) -> str:
    if state.get("compliance_blocked"):
        return "direct"
    return "intent"


async def resolve_intent_node(
    state: AgentState,
    llm: BaseChatModel,
    retriever: Any = None,
) -> dict:
    """
    企业级状态机：CheckSession → Gateway(Corpus) → Rewrite → IntentLLM → SlotValidate。
    """
    session_paper_ids = state.get("paper_ids") or []
    raw_query = (state.get("query") or "").strip()
    rewrite_reason = ""
    intent_confidence = 0.0
    gateway_reason = ""
    gateway_decision = None
    slot_memory = state.get("slot_memory")
    turn_id = state.get("turn_id") or ""

    # Phase 1: CheckSession
    if not session_paper_ids:
        placeholder = RetrievalIntent(
            effective_query=raw_query,
            intent="need_clarification",
            missing=[],
            confidence=0.0,
        )
        validated = resolve_turn(
            placeholder,
            raw_query,
            session_paper_ids,
            raw_query=raw_query,
            slot_memory=slot_memory,
            turn_id=turn_id,
        )
    else:
        # Phase 2: Gateway（Meta + Corpus Gate，Rewrite 之前）
        import asyncio

        from config import Config
        from rag.factory import EmbeddingService

        embeddings = None
        try:
            embeddings = EmbeddingService.get_embeddings(Config.EMBEDDING_MODEL)
        except Exception:
            pass

        gateway_decision = await asyncio.to_thread(
            run_gateway,
            raw_query,
            session_paper_ids,
            messages=state.get("messages", []),
            pending_user_query=state.get("pending_user_query", ""),
            summary=state.get("summary", ""),
            embeddings=embeddings,
        )
        gateway_reason = gateway_decision.reason

        trace = get_trace(state.get("trace_id"))
        if trace:
            trace.gateway_resolution(
                action=gateway_decision.action,
                reason=gateway_decision.reason,
                gate_query=gateway_decision.gate_query,
                content_score=gateway_decision.content_score,
                best_paper_id=gateway_decision.best_paper_id,
                latency_ms=gateway_decision.latency_ms,
                detail=gateway_decision.trace,
            )

        if gateway_decision.action != "continue":
            placeholder = RetrievalIntent(
                effective_query=raw_query,
                intent="need_clarification",
                missing=[],
                confidence=0.0,
            )
            validated = resolve_turn(
                placeholder,
                raw_query,
                session_paper_ids,
                raw_query=raw_query,
                scope_continue=False,
                slot_memory=slot_memory,
                turn_id=turn_id,
                rewrite_reason=rewrite_reason,
            )
        else:
            # Phase 3: Rewrite
            raw_query, effective_query, rewrite_reason = rewrite_query_for_intent(state)

            # Phase 4: IntentLLM（仅槽位理解）
            include_pending = rewrite_reason != "topic_switch"
            intent = await resolve_intent_llm(
                state,
                llm,
                effective_query=effective_query,
                include_pending=include_pending,
            )
            if intent.effective_query.strip() != effective_query:
                effective_query = intent.effective_query.strip() or effective_query
                intent.effective_query = effective_query
            intent_confidence = intent.confidence

            # Phase 5: SlotValidate
            validated = resolve_turn(
                intent,
                effective_query,
                session_paper_ids,
                raw_query=raw_query,
                scope_continue=True,
                slot_memory=slot_memory,
                turn_id=turn_id,
                rewrite_reason=rewrite_reason,
            )

    logger.info(
        "Turn: state=%s kind=%s mode=%s focus=%s clarify=%s direct=%s reason=%s",
        validated.get("turn_state"),
        validated.get("intent"),
        validated.get("scope_mode"),
        validated.get("focus_paper_ids"),
        validated.get("needs_clarification"),
        validated.get("direct_response"),
        validated.get("match_reason"),
    )
    trace = get_trace(state.get("trace_id"))
    if trace:
        if rewrite_reason:
            trace.query_rewrite(
                raw_query=raw_query,
                effective_query=validated["query"],
                reason=rewrite_reason,
            )
        trace.intent_resolution(
            intent=validated.get("intent", "paper_qa"),
            scope_mode=validated.get("scope_mode", ""),
            session_paper_ids=session_paper_ids,
            focus_paper_ids=validated.get("focus_paper_ids") or [],
            needs_clarification=validated.get("needs_clarification", False),
            match_reason=validated.get("match_reason", ""),
            candidate_paper_ids=validated.get("candidate_paper_ids") or [],
            retrieval_mode=validated.get("retrieval_mode", "body"),
            confidence=intent_confidence,
            missing_slots=validated.get("missing_slots") or [],
            gateway_reason=gateway_reason,
            content_score=(
                gateway_decision.content_score if gateway_decision else None
            ),
        )

    out: dict = dict(validated)
    out["turn_state"] = validated.get("turn_state", "")
    if gateway_decision is not None:
        out["gateway"] = {
            "action": gateway_decision.action,
            "reason": gateway_decision.reason,
            "gate_query": gateway_decision.gate_query,
            "content_score": gateway_decision.content_score,
            "best_paper_id": gateway_decision.best_paper_id,
        }
    if validated.get("direct_response"):
        out["pending_user_query"] = ""
    elif validated.get("needs_clarification"):
        out["pending_user_query"] = raw_query
    elif rewrite_reason in (
        "clarification_followup",
        "clarification+anaphoric",
        "topic_switch",
    ) or not validated.get("needs_clarification"):
        out["pending_user_query"] = ""
    return out


def respond_direct(state: AgentState) -> dict:
    """非 RAG 意图：静态回复，不进入检索。"""
    msg = state.get("answer") or ""
    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.clarification_response(msg)
    return {"answer": msg, "citations": []}


def respond_clarification(state: AgentState) -> dict:
    """多 PDF 消歧失败：直接返回提示，不进入检索。"""
    msg = state.get("clarification_message") or (
        "请指明要查询的论文（paper_id 或上传文件名）。"
    )
    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.clarification_response(msg)
    return {"answer": msg, "citations": []}


def route_after_intent(state: AgentState) -> str:
    """resolve_intent 之后：非 RAG 直出 / 消歧 / 检索分析。"""
    if state.get("direct_response"):
        return "direct"
    return "clarify" if state.get("needs_clarification") else "analyze"


def _infer_focus_paper_id(
    query: str,
    valid: list,
    session_paper_ids: list[str],
) -> str | None:
    """多 PDF 会话下，从子 Agent 检索结果推断「本题指向的一篇 paper_id」。"""
    if len(session_paper_ids) <= 1:
        return session_paper_ids[0] if session_paper_ids else None
    if COMPLEX_HINTS.search(query):
        return None

    session = set(session_paper_ids)
    counts: dict[str, int] = {}
    for sa in valid:
        for c in sa.get("citations") or []:
            pid = str(c.get("paper_id", ""))
            if pid in session:
                counts[pid] = counts.get(pid, 0) + 1
    if not counts:
        return None

    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    top_pid, top_n = ranked[0]
    second_n = ranked[1][1] if len(ranked) > 1 else 0
    if top_n > second_n:
        return top_pid
    if _SINGLE_PAPER_QUERY_HINTS.search(query):
        return top_pid
    return None


def _remap_answer_citations_to_final(
    answer: str,
    sa_citations: list[dict],
    chunk_to_idx: dict[str, int],
) -> str:
    """把子答案里的局部 [n] 映射为 dedupe 后 citation 列表的全局 [n]。"""

    def _map_local(local_n: int) -> int | None:
        if 1 <= local_n <= len(sa_citations):
            cid = str(sa_citations[local_n - 1].get("chunk_id", ""))
            return chunk_to_idx.get(cid) if cid else None
        return None

    def _replace_bracket(m: re.Match[str]) -> str:
        mapped: list[str] = []
        for part in re.split(r"[,，\s]+", m.group(1).strip()):
            if part.isdigit():
                idx = _map_local(int(part))
                if idx is not None:
                    mapped.append(str(idx))
        if mapped:
            return "[" + ", ".join(mapped) + "]"
        return m.group(0)

    return re.sub(r"\[([^\]]+)\]", _replace_bracket, answer)


def _format_failed_sub_queries(sub_answers: list) -> tuple[str, list[dict]]:
    """从 sub_answers 提取失败子问题，供合成提示与 SSE warnings 使用。"""
    failed = [
        {"query": sa["query"], "error": sa.get("error") or "unknown error"}
        for sa in sub_answers
        if is_failed_sub_answer(sa)
    ]
    if not failed:
        return "", []
    lines = [f"- {item['query']}: {item['error']}" for item in failed]
    note = (
        "\n# Unavailable Evidence\n"
        "The following sub-questions could NOT be retrieved (network/timeout). "
        "Do NOT invent facts for them; answer from available evidence only. "
        "If the user question depends mainly on a failed sub-question, state briefly "
        "that part of the retrieval was unavailable.\n"
        + "\n".join(lines)
    )
    return note, failed


def _chunk_index_map(citations: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, c in enumerate(citations, 1):
        cid = str(c.get("chunk_id", ""))
        if cid:
            out[cid] = i
    return out


def _format_citation_index(citations: list[dict]) -> str:
    if not citations:
        return "(no citations)"
    return "\n".join(
        f"[{i}] {CitationExtractor.format_citation(c)}"
        for i, c in enumerate(citations, 1)
    )


def _synthesis_fallback(state: AgentState) -> str:
    """合成只输出 [n] 时，回退到最相关子答案（引用已映射到全局编号）。"""
    valid = _collect_valid_evidence(_sub_answers_for_turn(state))
    if not valid:
        return ""
    query = sanitize_query(state["query"])
    sa = next((x for x in valid if x.get("query") == query), valid[0])
    final_citations = state.get("citations") or []
    return _remap_answer_citations_to_final(
        str(sa.get("answer", "")),
        sa.get("citations") or [],
        _chunk_index_map(final_citations),
    )


async def analyze_query(state: AgentState, llm: BaseChatModel) -> dict:
    """
    查询分解：将复杂问题拆成多个可独立检索的子查询。
    失败时退化为仅使用原始 query。
    """
    query = sanitize_query(state["query"])
    msgs = [SystemMessage(content=QUERY_ANALYZER), HumanMessage(content=query)]

    structured_llm = llm.with_structured_output(QueryAnalysis)
    try:
        result = await structured_llm.ainvoke(msgs)
        parsed = cast(QueryAnalysis, result)
        complexity = parsed.complexity if parsed.complexity in ("simple", "complex") else "simple"
        sub_queries = parsed.sub_queries
    except Exception:
        complexity = "simple"
        sub_queries = [query]

    hint = _heuristic_complexity(query)
    if hint:
        complexity = hint

    sub_queries = _normalize_sub_queries(query, sub_queries, complexity)

    logger.info(f"Decomposed complexity={complexity} into {len(sub_queries)} sub-queries")
    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.analyze(sub_queries, complexity=complexity)
    return {"sub_queries": sub_queries, "query_complexity": complexity}


def prepare_synthesis(state: AgentState) -> dict:
    """汇总有效子答案证据，构建 synthesize 消息（过滤 insufficient / 空答案 / 硬失败）。"""
    sub_answers = _sub_answers_for_turn(state)
    valid = _collect_valid_evidence(sub_answers)
    failure_note, failed_sub_queries = _format_failed_sub_queries(sub_answers)
    session_paper_ids = state.get("paper_ids") or []

    pooled: list[dict] = []
    for sa in valid:
        pooled.extend(sa.get("citations") or [])

    focus_ids = state.get("focus_paper_ids") or []
    if len(focus_ids) == 1:
        focus_paper = focus_ids[0]
    else:
        focus_paper = _infer_focus_paper_id(state["query"], valid, session_paper_ids)
    focus_applied = False
    if focus_paper:
        filtered = [c for c in pooled if c.get("paper_id") == focus_paper]
        if filtered:
            pooled = filtered
            focus_applied = True
            logger.info("Synthesis focus paper: %s (multi-PDF session)", focus_paper)

    final_citations = _dedupe_citations(pooled)
    chunk_to_idx = _chunk_index_map(final_citations)

    context_parts = []
    for i, sa in enumerate(valid, 1):
        sa_cites = sa.get("citations") or []
        if focus_applied:
            sa_cites = [c for c in sa_cites if c.get("paper_id") == focus_paper]
            if not sa_cites:
                continue
        remapped = _remap_answer_citations_to_final(
            str(sa.get("answer", "")),
            sa_cites,
            chunk_to_idx,
        )
        context_parts.append(
            f"### Evidence {i}\nSub-question: {sa['query']}\nFindings: {remapped}"
        )

    sub_context = "\n\n".join(context_parts) if context_parts else "(no valid evidence)"
    focus_note = ""
    if focus_applied:
        focus_note = (
            f"\n# Scope\nThis question targets paper `{focus_paper}`; "
            "cite only indices for that paper."
        )

    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.prepare_synthesis(sub_answers)

    system_content = SYNTHESIZER.format(
        citation_index=_format_citation_index(final_citations),
        context=sub_context,
        focus_note=focus_note,
        failure_note=failure_note,
    )

    user_lang = query_language(state["query"])
    lang_hint = "Respond in Chinese." if user_lang == "zh" else "Respond in English."

    return {
        "synth_messages": [
            SystemMessage(content=system_content),
            HumanMessage(
                content=f"Original question: {state['query']}\n{lang_hint}"
            ),
        ],
        "citations": final_citations,
        "failed_sub_queries": failed_sub_queries,
    }


async def synthesize(state: AgentState, llm: BaseChatModel) -> dict:
    """流式调用 LLM 生成最终综合答案（chat 路由通过 messages 流捕获 token）。"""
    synth_messages = state.get("synth_messages", [])
    if not synth_messages:
        return {"answer": ""}

    failed = state.get("failed_sub_queries") or []
    valid = _collect_valid_evidence(_sub_answers_for_turn(state))
    lang = query_language(state.get("query", ""))
    if not valid:
        if failed:
            if lang == "zh":
                answer = (
                    "部分子问题检索未完成（可能超时或服务暂时不可用），"
                    "当前没有足够的文献证据生成可靠答案。请稍后重试或缩小问题范围。"
                )
            else:
                answer = (
                    "Some sub-query retrievals did not complete; "
                    "there is not enough evidence to produce a reliable answer. "
                    "Please retry or narrow your question."
                )
        elif lang == "zh":
            answer = "未检索到与问题相关的文献片段，请确认论文已上传或尝试更具体的问题。"
        else:
            answer = "No relevant passages were retrieved. Check uploads or try a more specific question."
        trace = get_trace(state.get("trace_id"))
        if trace:
            trace.synthesize(answer)
        return {"answer": answer}

    async def _collect_stream() -> str:
        chunks: list[str] = []
        async for chunk in llm.astream(synth_messages):
            text = message_content_to_str(chunk.content)
            if text:
                chunks.append(text)
        return "".join(chunks)

    try:
        answer = await asyncio.wait_for(
            _collect_stream(),
            timeout=effective_timeout(float(Config.SYNTHESIZE_STEP_TIMEOUT)),
        )
    except asyncio.TimeoutError:
        logger.warning("synthesize timed out after %ss", Config.SYNTHESIZE_STEP_TIMEOUT)
        answer = _synthesis_fallback(state) or (
            "回答生成超时，请稍后重试。"
            if query_language(state.get("query", "")) == "zh"
            else "Answer generation timed out; please retry."
        )
    if is_citation_only_answer(answer):
        fallback = _synthesis_fallback(state)
        if fallback:
            logger.warning("Synthesizer returned citation-only answer; using sub-answer fallback")
            answer = fallback

    final_citations = state.get("citations") or []
    if answer and final_citations:
        answer, stripped = sanitize_answer_citations(answer, len(final_citations))
        if stripped:
            logger.warning(
                "Stripped invalid synthesis citation indices: %s (max=%d)",
                stripped,
                len(final_citations),
            )

    grounding_verdict = None
    if answer and final_citations:
        grounding_verdict = check_grounding(answer, final_citations)
        if not grounding_verdict.ok:
            logger.warning(
                "Grounding check failed: reason=%s coverage=%.2f",
                grounding_verdict.reason,
                grounding_verdict.citation_coverage,
            )
            if grounding_verdict.reason in (
                "citations_but_no_brackets",
                "low_citation_coverage",
                "many_invalid_citations",
            ):
                fallback = _synthesis_fallback(state)
                if fallback:
                    fallback, _ = sanitize_answer_citations(
                        fallback, len(final_citations)
                    )
                    fallback_verdict = check_grounding(fallback, final_citations)
                    if fallback_verdict.ok:
                        logger.info(
                            "Grounding check recovered via sub-answer fallback "
                            "(coverage=%.2f)",
                            fallback_verdict.citation_coverage,
                        )
                        answer = fallback
                        grounding_verdict = fallback_verdict
                if not grounding_verdict.ok:
                    if lang == "zh":
                        answer = (
                            "根据检索到的证据，无法生成有足够引文支撑的回答。"
                            "请尝试更具体的问题。"
                        )
                    else:
                        answer = (
                            "The retrieved evidence does not support a well-cited answer. "
                            "Please try a more specific question."
                        )

    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.synthesize(answer)
        if grounding_verdict is not None:
            trace.grounding_check(
                ok=grounding_verdict.ok,
                citation_coverage=grounding_verdict.citation_coverage,
                reason=grounding_verdict.reason,
                stripped_indices=grounding_verdict.stripped_indices,
            )
    return {"answer": answer}


async def summarize_conversation(state: AgentState, llm: BaseChatModel) -> dict:
    """
    对话记忆压缩：消息数超过 WINDOW_SIZE 时，将更早消息摘要并 RemoveMessage 删除。
    """
    messages = state.get("messages", [])
    existing_summary = state.get("summary", "")

    if len(messages) <= WINDOW_SIZE:
        return {}

    to_summarize = messages[:-WINDOW_SIZE]

    lines = []
    if existing_summary:
        lines.append(f"Previous summary: {existing_summary}")
    for msg in to_summarize:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            continue
        lines.append(f"{role}: {msg.content}")

    if not lines:
        return {}

    history = "\n".join(lines)

    resp = await llm.ainvoke([
        SystemMessage(content=SUMMARIZER.format(history=history)),
        HumanMessage(content="Summarize the above conversation."),
    ])

    return {
        "summary": resp.content,
        "messages": [RemoveMessage(id=m.id) for m in to_summarize if m.id],
    }


def _boost_docs_by_query_tokens(
    query: str,
    docs: list[Document],
    session_paper_ids: list[str],
) -> list[Document]:
    """session_wide 检索时，若 query 英文 token 唯一匹配某 paper_id，前置该论文段落。"""
    q_tokens = {t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", query)}
    if not q_tokens:
        return docs
    scores: dict[str, int] = {}
    for pid in session_paper_ids:
        pid_tokens = set(pid.lower().replace("-", "_").split("_"))
        scores[pid] = len(q_tokens & pid_tokens)
    best = max(scores.values()) if scores else 0
    if best <= 0:
        return docs
    winners = [pid for pid, s in scores.items() if s == best]
    if len(winners) != 1:
        return docs
    return boost_docs_for_focus_paper(docs, winners[0])


# --------------- 子图节点（单个子查询：retrieve → generate → reflect → 可能 retry）---------------


async def retrieve(state: SubAgentState, retriever, citation_extractor) -> dict:
    """
    【子图第 1 步：检索】

    从 Milvus 拉取与当前子问题相关的论文片段，并写入 state：
    - documents：给 LLM 看的上下文字符串（带 Source）
    - citations：给前端展示的引用元数据
    - needs_vlm：下一步 generate 是否要主动看图（VLM）

    入参 retriever：实际是 dependencies.RetrieverTool，内部调用混合检索 + 重排。
    入参 citation_extractor：通常是 CitationExtractor 类，用于抽 paper/章节/页码。
    """
    from rag.factory import is_visual_query  # 判断用户问题是否「明显在问图/表」

    # 当前子 Agent 要回答的这一条子问题（主图 analyze 拆分后的其中一条）
    query = sanitize_query(state["query"])
    # 若上一轮 reflect 认为答案不足，会填入补充检索词；首次检索时为空列表
    retry_queries = state.get("retry_queries", [])
    retrieval_mode = state.get("retrieval_mode", "body")

    queries = retry_queries if retry_queries else [query]
    focus_paper_ids = state.get("focus_paper_ids") or []
    if not retry_queries and retrieval_mode == "body":
        queries = _retrieval_query_variants(
            query, queries, focus_paper_ids=focus_paper_ids,
        )

    node_type_filter = ["paper_profile"] if retrieval_mode == "profile" else None
    section_type_filter = None
    session_paper_ids = state.get("paper_ids") or []
    focus_paper_ids = state.get("focus_paper_ids") or []
    effective_ids = effective_retrieval_paper_ids(session_paper_ids, focus_paper_ids)
    paper_id_filter = effective_ids if effective_ids else None
    if paper_id_filter:
        logger.info(
            "Retrieve scope: paper_ids=%s (focus=%s mode=%s)",
            paper_id_filter,
            focus_paper_ids or "—",
            "explicit" if focus_paper_ids else "session",
        )

    def _invoke(q: str) -> list[Document]:
        """同步检索一次（可重试；阻塞故放线程池）。"""

        def _do() -> list[Document]:
            return retriever.invoke(
                q,
                section_type_filter=section_type_filter,
                paper_id_filter=paper_id_filter,
                node_type_filter=node_type_filter,
            )

        trace_local = get_trace(state.get("trace_id"))

        def _on_retry(attempt: int, exc: BaseException) -> None:
            if trace_local:
                trace_local.retry_attempt(
                    operation="retrieve",
                    attempt=attempt,
                    max_attempts=Config.RETRIEVE_MAX_RETRIES,
                    error=f"{type(exc).__name__}: {exc}",
                    sub_query=query,
                    search_query=q,
                )

        return retry_sync(
            _do,
            label=f"retrieve({q[:30]!r})",
            max_attempts=Config.RETRIEVE_MAX_RETRIES,
            backoff_ms=_RETRIEVE_BACKOFF_MS,
            on_retry=_on_retry,
        )

    per_query_timeout = effective_timeout(Config.RETRIEVE_STEP_TIMEOUT)
    loop = asyncio.get_event_loop()

    async def _retrieve_one(q: str) -> list[Document]:
        async with _RETRIEVE_SEM:
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, _invoke, q),
                    timeout=per_query_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Retrieve query timed out after %.0fs: %r",
                    per_query_timeout,
                    q[:60],
                )
                trace_local = get_trace(state.get("trace_id"))
                if trace_local:
                    trace_local.retry_attempt(
                        operation="retrieve_step_timeout",
                        attempt=1,
                        max_attempts=1,
                        error=f"query timed out after {per_query_timeout:.0f}s",
                        sub_query=query,
                        search_query=q,
                    )
                return []

    results = await asyncio.gather(*[_retrieve_one(q) for q in queries])

    trace = get_trace(state.get("trace_id"))
    if trace:
        for q, batch in zip(queries, results):
            trace.retrieval(
                sub_query=query,
                search_queries=[q],
                section_type_filter=section_type_filter,
                paper_id_filter=paper_id_filter,
                node_type_filter=node_type_filter,
                retrieval_mode=retrieval_mode,
                docs=batch,
            )

    seen_ids: set[str] = set()  # 已见过的 chunk_id，用于去重
    docs: list[Document] = []   # 合并去重后的 Document 列表
    for batch in results:       # 遍历每个 query 的检索批次
        for doc in batch:       # 遍历该批次中的每个文档块
            cid = doc.metadata.get("chunk_id", id(doc))  # 块唯一 ID，没有则用 Python 对象 id
            if cid not in seen_ids:  # 同一 chunk 只保留一份（多 query 可能命中相同段落）
                seen_ids.add(cid)
                docs.append(doc)

    if len(focus_paper_ids) == 1:
        docs = boost_docs_for_focus_paper(docs, focus_paper_ids[0])
    elif len(session_paper_ids) > 1 and not focus_paper_ids:
        docs = _boost_docs_by_query_tokens(query, docs, session_paper_ids)

    verdict = evaluate_evidence(docs, query)
    if not verdict.passed:
        logger.warning(
            "Evidence gate: signal=%s top=%.3f reason=%s — treating as no evidence",
            verdict.signal,
            verdict.top_score,
            verdict.reason,
        )
        docs = []
    if trace:
        trace.evidence_gate(
            passed=verdict.passed,
            top_score=verdict.top_score,
            signal=verdict.signal,
            reason=verdict.reason,
            sub_query=query,
        )

    # 从 Document 抽出引用信息（paper_id、section、page、node_type、完整 metadata）
    citations = citation_extractor.extract_all(docs) if docs else []
    documents = []  # 注意：变量名 documents 这里是「字符串列表」，不是 LangChain Document
    for doc, cite in zip(docs, citations):  # 正文与引用一一对应
        source = citation_extractor.format_citation(cite)  # 如 "Paper: xxx | Section: ... | Page: 3"
        # 拼成 LLM 上下文：正文 + 来源行（generate 里会再加 [1][2] 编号）
        documents.append(f"{doc.page_content}\n[Source: {source}]")

    if len(docs) != len(citations):
        logger.warning(f"Doc/citation count mismatch: {len(docs)} docs vs {len(citations)} citations")

    # 检索结果里是否包含「带裁剪图路径」的 figure 节点（有图才可能走 VLM）
    has_figure = any(
        c.get("node_type") == "figure" and c.get("metadata", {}).get("image_path")
        for c in citations
    )
    # 仅当：用户问题像在看图 + 确实检出了图 → 下一步 generate 标记需要 VLM
    needs_vlm = is_visual_query(query) and has_figure

    truncated = query[:50] + ("..." if len(query) > 50 else "")  # 日志里缩短显示
    logger.info(f"Retrieved {len(documents)} docs for: {truncated} (queries={len(queries)}) | needs_vlm={needs_vlm}")
    if trace:
        trace.retrieval(
            sub_query=query,
            search_queries=queries,
            section_type_filter=section_type_filter,
            paper_id_filter=paper_id_filter,
            docs=docs,
            merged_total=len(docs),
        )
    # 返回 dict 会合并进 SubAgentState，供 generate / reflect 使用
    return {"documents": documents, "citations": citations, "needs_vlm": needs_vlm}


async def generate(state: SubAgentState, llm: BaseChatModel, vision_service=None) -> dict:
    """
    【子图第 2 步：生成答案】

    用 retrieve 得到的 documents 作为上下文，让 LLM 按 GENERATOR 提示词作答（须带 [n] 引用）。
    若 retrieve 设置了 needs_vlm，且配置了 vision_service，则先让 VLM 读图再把描述塞进上下文。
    """
    query = state["query"]  # 子问题原文
    documents = state.get("documents", [])  # 字符串列表，每条是一段检索文本 + Source
    citations = state.get("citations", [])  # 结构化引用，含 metadata.image_path 等
    needs_vlm = state.get("needs_vlm", False)  # retrieve 节点是否建议走视觉增强

    if not documents:
        # 向量库无命中或过滤后为空，直接返回固定话术（不再调 LLM）
        trace = get_trace(state.get("trace_id"))
        if trace:
            trace.generate(sub_query=query, context_count=0, answer="No relevant information found.")
        return {"answer": "No relevant information found.", "citations": []}

    # ---------- 主动 VLM 路径（retrieve 已判定 needs_vlm）----------
    if needs_vlm and vision_service:  # vision_service 为 None 时跳过（未配置 VLM）
        from rag.factory import should_invoke_vlm  # 二次确认是否值得调 VLM（省成本）
        has_figure = any(
            c.get("node_type") == "figure" and c.get("metadata", {}).get("image_path")
            for c in citations
        )
        if should_invoke_vlm(query, has_figure):  # 默认：视觉类问题 + 有图 → True
            for cite in citations:  # 遍历每条引用
                if cite.get("node_type") != "figure":  # 只处理图节点
                    continue
                image_path = cite.get("metadata", {}).get("image_path")  # PyMuPDF 裁出的 PNG 路径
                if not image_path:
                    continue  # 无图文件则跳过
                vlm_desc = cite.get("metadata", {}).get("vlm_description")  # 是否已分析过（缓存）
                if not vlm_desc:
                    caption = cite.get("text", "")  # 图注文字（可能为空）
                    # 调多模态模型：读图 + 返回文字描述
                    vlm_desc = vision_service.analyze_figure(image_path, caption)
                    if vlm_desc:
                        # 写回 metadata，同一会话内 reflect 兜底时不必重复调 VLM
                        cite.setdefault("metadata", {})["vlm_description"] = vlm_desc
                if vlm_desc:
                    # 把 VLM 描述当作额外上下文段落，前缀便于 LLM 识别
                    documents = list(documents) + [f"[Figure Analysis] {vlm_desc}"]
                    logger.info(f"VLM analysis injected for figure: {image_path}")

    # 给每段上下文编号 [1]、[2]...，与 GENERATOR 提示里「用 [i] 引用」一致
    context = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(documents, 1))
    if len(context) > MAX_GENERATE_CONTEXT_CHARS:
        context = context[:MAX_GENERATE_CONTEXT_CHARS] + "\n\n[Context truncated due to length limit.]"

    resp = await _invoke_llm_with_retry(
        llm,
        [
            SystemMessage(content=GENERATOR),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ],
        operation="generate",
        sub_query=query,
        trace_id=state.get("trace_id", ""),
        step_timeout=float(Config.GENERATE_STEP_TIMEOUT),
    )

    answer = _sanitize_generate_answer(str(resp.content), context)
    answer = _ensure_primary_citation(answer, documents, query)
    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.generate(
            sub_query=query,
            context_count=len(documents),
            answer=answer,
            needs_vlm=needs_vlm,
        )
    return {"answer": answer}  # 写入 state["answer"]，供 reflect 评估


async def reflect(state: SubAgentState, llm: BaseChatModel, vision_service=None) -> dict:
    """
    【子图第 3 步：反思 / 质量检查】

    用 REFLECTOR 提示词让 LLM 判断：当前 answer 是否充分回答问题。
    - 充分 → is_sufficient=True，子图结束（或走 done 边）
    - 不足 → 给出 retry_queries，配合 should_retry 回到 retrieve 再搜一轮

    额外「VLM 兜底」：文本答不好但检索里有图、且还没用过 VLM 时，
    在 reflect 里直接调 VLM + 重新 generate，并强制 is_sufficient=True（不再 retry）。
    """
    query = state["query"]
    answer = state.get("answer", "")       # generate 刚产出的答案
    documents = state.get("documents", [])  # 当前上下文（字符串列表）
    citations = state.get("citations", [])
    retries = state.get("retries", 0)     # 已经反思/重试过的次数（每进一次 reflect +1）

    if not documents:
        return {"is_sufficient": True, "retry_queries": [], "retries": retries + 1}

    if _rule_reflect_sufficient(answer):
        trace = get_trace(state.get("trace_id"))
        if trace:
            trace.reflect(
                sub_query=query,
                is_sufficient=True,
                retry_queries=[],
                retries=retries + 1,
                answer_preview=str(answer),
                skipped_reason="rule_sufficient",
            )
        return {"is_sufficient": True, "retry_queries": [], "retries": retries + 1}
    if _is_insufficient_answer(answer):
        trace = get_trace(state.get("trace_id"))
        if trace:
            trace.reflect(
                sub_query=query,
                is_sufficient=True,
                retry_queries=[],
                retries=retries + 1,
                answer_preview=str(answer),
                skipped_reason="insufficient_boilerplate",
            )
        return {"is_sufficient": True, "retry_queries": [], "retries": retries + 1}

    # 约束 LLM 输出为 ReflectionResult（is_sufficient + retry_queries）
    structured_llm = llm.with_structured_output(ReflectionResult)
    try:
        result = await _invoke_llm_with_retry(
            structured_llm,
            [
                SystemMessage(content=REFLECTOR),
                HumanMessage(content=f"Question: {query}\nAnswer: {answer}"),
            ],
            operation="reflect",
            sub_query=query,
            trace_id=state.get("trace_id", ""),
            step_timeout=float(Config.REFLECT_STEP_TIMEOUT),
        )
        reflection = cast(ReflectionResult, result)
        is_sufficient = reflection.is_sufficient
        retry_queries = reflection.retry_queries
    except Exception:
        is_sufficient = True
        retry_queries = []

    if _rule_reflect_sufficient(answer):
        is_sufficient = True
        retry_queries = []
    elif _is_insufficient_answer(answer):
        is_sufficient = True
        retry_queries = []

    # ---------- VLM 兜底路径（与 generate 里主动 VLM 不同：这里是「答不好再看图」）----------
    if not is_sufficient and vision_service and not state.get("needs_vlm", False):
        # 条件：答案不足 + 有 VLM 服务 + retrieve 阶段未走过主动 VLM（避免重复花钱）
        from rag.factory import should_invoke_vlm
        figure_citations = [
            c for c in citations
            if c.get("node_type") == "figure" and c.get("metadata", {}).get("image_path")
        ]
        has_figure = bool(figure_citations)
        # 传入 answer：若答案里已有「信息不足」等话术，也会触发 VLM
        if should_invoke_vlm(query, has_figure, answer=answer):
            extra_docs = []  # 待追加的 VLM 描述段落
            for cite in figure_citations[:2]:  # 最多处理 2 张图，控制 API 成本
                image_path = cite["metadata"]["image_path"]
                vlm_desc = cite["metadata"].get("vlm_description")
                if not vlm_desc:
                    caption = cite.get("text", "")
                    vlm_desc = vision_service.analyze_figure(image_path, caption)
                    if vlm_desc:
                        cite["metadata"]["vlm_description"] = vlm_desc
                if vlm_desc:
                    extra_docs.append(f"[Figure Analysis] {vlm_desc}")
                    logger.info(f"VLM fallback triggered for: {image_path}")

            if extra_docs:  # 成功拿到至少一段图描述
                enhanced_docs = list(documents) + extra_docs  # 原文档上下文 + 图分析
                context = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(enhanced_docs, 1))
                # 在 reflect 内直接第二次生成（不再走 generate 节点）
                resp = await _invoke_llm_with_retry(
                    llm,
                    [
                        SystemMessage(content=GENERATOR),
                        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
                    ],
                    operation="generate_vlm_fallback",
                    sub_query=query,
                    trace_id=state.get("trace_id", ""),
                    step_timeout=float(Config.GENERATE_STEP_TIMEOUT),
                )
                vlm_answer = resp.content
                trace = get_trace(state.get("trace_id"))
                if trace:
                    trace.generate(
                        sub_query=query,
                        context_count=len(enhanced_docs),
                        answer=str(vlm_answer),
                        needs_vlm=True,
                    )
                    trace.reflect(
                        sub_query=query,
                        is_sufficient=True,
                        retry_queries=[],
                        retries=retries + 1,
                        answer_preview=str(vlm_answer),
                    )
                return {
                    "answer": vlm_answer,           # 覆盖原 answer
                    "documents": enhanced_docs,       # 更新上下文
                    "needs_vlm": True,                # 标记已用 VLM，防止再次兜底
                    "is_sufficient": True,            # 强制结束反思循环
                    "retry_queries": [],              # 不再走文本重检索
                    "retries": retries + 1,
                }

    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.reflect(
            sub_query=query,
            is_sufficient=is_sufficient,
            retry_queries=retry_queries,
            retries=retries + 1,
            answer_preview=str(answer),
        )

    # 常规返回：由 graph 的 should_retry 决定走 retry 还是 END
    return {
        "is_sufficient": is_sufficient,
        "retry_queries": retry_queries,  # 非空时 prepare_retry 会取第一条作为新 query
        "retries": retries + 1,          # 反思次数 +1（与 max_retries 比较）
    }


# --------------- 子图路由（LangGraph 条件边用的函数）---------------


def should_retry(state: SubAgentState, max_retries: int = 2) -> str:
    """
    【条件路由】reflect 之后调用，返回字符串决定下一节点：

    - "retry" → prepare_retry → retrieve（用新的/补充 query 再搜）
    - "done"  → 子图结束，把 SubAnswer 汇总回主图

    max_retries 来自 Config.MAX_RETRIES，默认 2。
    """
    # 不足 且 反思次数未超限 → 重试；否则结束子图
    if not state.get("is_sufficient", True) and state.get("retries", 0) < max_retries:
        logger.info(f"Reflection: insufficient, retrying ({state.get('retries', 0)}/{max_retries})")
        return "retry"
    return "done"


def prepare_retry(state: SubAgentState) -> dict:
    """
    【重试准备节点】在再次进入 retrieve 之前执行。

    把 reflect 给出的 retry_queries[0] 写回 state["query"]，
    这样下一轮 retrieve 会用「更针对缺失信息」的 query 去搜。
    若 retry_queries 为空则仍用原子问题。
    """
    retry_queries = state.get("retry_queries", [])
    query = retry_queries[0] if retry_queries else state["query"]
    return {"query": query}  # 仅更新 query 字段，retry_queries 仍留在 state（retrieve 会读全列表）
