"""
LangGraph 节点函数实现。

分两类：
1. 主图节点：summarize / classify / analyze / prepare_synthesis / synthesize
2. 子图节点：retrieve / generate / reflect / prepare_retry / should_retry

依赖 Pydantic 结构化输出（QueryAnalysis、QueryClassification、ReflectionResult）
保证 LLM 返回可解析 JSON。
"""

import re
import logging
from typing import cast

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document

from .states import AgentState, SubAgentState
from .prompts import QUERY_ANALYZER, QUERY_CLASSIFIER, SYNTHESIZER, GENERATOR, REFLECTOR, SUMMARIZER
from rag.chat_trace import get_trace
from rag.citation import CitationExtractor, is_citation_only_answer

logger = logging.getLogger(__name__)

# 送入本地 LLM 的检索上下文上限，避免 Ollama 因 prompt 过长返回 502
MAX_GENERATE_CONTEXT_CHARS = 12_000

# 保留最近 N 条消息原文；更早的由 summarize 压缩进 summary
WINDOW_SIZE = 6

_INSUFFICIENT_PHRASES = (
    "does not contain sufficient information",
    "insufficient information",
    "信息不足",
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
_COMPLEX_HINTS = re.compile(
    r"对比|比较|分别|差异|以及.+和|vs\.?|compare|respectively|both .+ and",
    re.I,
)
_ABSTRACT_HINTS = re.compile(r"摘要|abstract", re.I)
_SINGLE_PAPER_QUERY_HINTS = re.compile(
    r"这篇|该论文|该篇|此文|本文|this paper|the paper|摘要里|摘要中|摘要采用|摘要提到|摘要表示|"
    r"abstract says|in the abstract",
    re.I,
)

MAX_SUB_QUERIES_SIMPLE = 2
MAX_SUB_QUERIES_COMPLEX = 4


def _is_insufficient_answer(text: str) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in _INSUFFICIENT_PHRASES)


def _query_language(query: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"


def sanitize_query(query: str) -> str:
    """去掉前端/JSON 误入的尾部引号与反斜杠。"""
    q = (query or "").strip()
    q = re.sub(r'\\+"\s*$', "", q)
    q = re.sub(r'"\s*$', "", q)
    return q.strip()


def _heuristic_complexity(query: str) -> str | None:
    if _COMPLEX_HINTS.search(query):
        return "complex"
    if _SIMPLE_HINTS.search(query):
        return "simple"
    return None


def _normalize_sub_queries(query: str, sub_queries: list[str], complexity: str) -> list[str]:
    lang = _query_language(query)
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
    return out[:max_n]


def _retrieval_query_variants(query: str, queries: list[str]) -> list[str]:
    """摘要类问题追加检索词，提升命中 摘要/Abstract 段落。"""
    if not _ABSTRACT_HINTS.search(query):
        return queries
    extra = []
    if _query_language(query) == "zh":
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


class QueryClassification(BaseModel):
    """classify_query 的结构化输出。"""
    query_type: str = Field(description="One of: experimental_result, method, background, general")


class ReflectionResult(BaseModel):
    """reflect 节点的结构化输出。"""
    is_sufficient: bool = Field(description="Whether the answer is sufficient")
    retry_queries: list[str] = Field(default_factory=list, description="Queries for missing info")


def _build_context_header(state: AgentState) -> str:
    """
    为多轮对话构建 XML 风格上下文块：摘要 + 最近 WINDOW_SIZE 轮对话。
    供 analyze / prepare_synthesis 等节点注入系统提示。
    """
    parts = []
    summary = state.get("summary", "")
    if summary:
        parts.append(f"<conversation_summary>\n{summary}\n</conversation_summary>")

    recent = state.get("messages", [])[-WINDOW_SIZE:]
    if recent:
        lines = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                continue
            lines.append(f"{role}: {msg.content}")
        if lines:
            parts.append("<recent_conversation>\n" + "\n".join(lines) + "\n</recent_conversation>")

    return "\n\n".join(parts)


# --------------- 主图节点 ---------------


def _infer_focus_paper_id(
    query: str,
    valid: list,
    session_paper_ids: list[str],
) -> str | None:
    """多 PDF 会话下，从子 Agent 检索结果推断「本题指向的一篇 paper_id」。"""
    if len(session_paper_ids) <= 1:
        return session_paper_ids[0] if session_paper_ids else None
    if _COMPLEX_HINTS.search(query):
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
    context_header = _build_context_header(state)

    system_content = QUERY_ANALYZER
    if context_header:
        system_content += f"\n\n# Conversation Context\n{context_header}"
    msgs = [SystemMessage(content=system_content), HumanMessage(content=query)]

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


async def classify_query(state: AgentState, llm: BaseChatModel) -> dict:
    """
    查询分类：决定检索时是否按 section_type（experiment/method/background）过滤。
    同时写入 ContextVar 供 section_type 路由使用。
    """
    from .tools import set_query_type
    query = sanitize_query(state["query"])

    # 摘要/abstract 题：入库 chunk 多为 section_type=other，勿用 experiment 过滤
    if _ABSTRACT_HINTS.search(query):
        query_type = "general"
        set_query_type(query_type)
        logger.info("Query classified as: %s (abstract hint override)", query_type)
        trace = get_trace(state.get("trace_id"))
        if trace:
            trace.classify(query_type)
        return {"query_type": query_type}

    structured_llm = llm.with_structured_output(QueryClassification)
    try:
        result = await structured_llm.ainvoke([
            SystemMessage(content=QUERY_CLASSIFIER),
            HumanMessage(content=query),
        ])
        query_type = cast(QueryClassification, result).query_type
        if query_type not in ("experimental_result", "method", "background", "general"):
            query_type = "general"
    except Exception:
        query_type = "general"

    set_query_type(query_type)
    logger.info(f"Query classified as: {query_type}")
    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.classify(query_type)
    return {"query_type": query_type}


def prepare_synthesis(state: AgentState) -> dict:
    """汇总有效子答案证据，构建 synthesize 消息（过滤 insufficient / 空答案）。"""
    sub_answers = _sub_answers_for_turn(state)
    valid = _collect_valid_evidence(sub_answers)
    context_header = _build_context_header(state)
    session_paper_ids = state.get("paper_ids") or []

    pooled: list[dict] = []
    for sa in valid:
        pooled.extend(sa.get("citations") or [])

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
    )
    if context_header:
        system_content += f"\n\n# Conversation Context\n{context_header}"

    user_lang = _query_language(state["query"])
    lang_hint = "Respond in Chinese." if user_lang == "zh" else "Respond in English."

    return {
        "synth_messages": [
            SystemMessage(content=system_content),
            HumanMessage(
                content=f"Original question: {state['query']}\n{lang_hint}"
            ),
        ],
        "citations": final_citations,
    }


async def synthesize(state: AgentState, llm: BaseChatModel) -> dict:
    """流式调用 LLM 生成最终综合答案（chat 路由通过 messages 流捕获 token）。"""
    synth_messages = state.get("synth_messages", [])
    if not synth_messages:
        return {"answer": ""}
    chunks = []
    async for chunk in llm.astream(synth_messages):
        if chunk.content:
            chunks.append(chunk.content)
    answer = "".join(chunks)
    if is_citation_only_answer(answer):
        fallback = _synthesis_fallback(state)
        if fallback:
            logger.warning("Synthesizer returned citation-only answer; using sub-answer fallback")
            answer = fallback
    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.synthesize(answer)
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
    import asyncio  # 用于并发执行多次检索（反思重试时会有多个 query）

    # 当前子 Agent 要回答的这一条子问题（主图 analyze 拆分后的其中一条）
    query = sanitize_query(state["query"])
    # 若上一轮 reflect 认为答案不足，会填入补充检索词；首次检索时为空列表
    retry_queries = state.get("retry_queries", [])
    # 主图 classify 节点写入：experimental_result / method / background / general
    query_type = state.get("query_type", "general")

    from .tools import SECTION_TYPE_ROUTE

    queries = retry_queries if retry_queries else [query]
    if not retry_queries:
        queries = _retrieval_query_variants(query, queries)

    section_type_filter = SECTION_TYPE_ROUTE.get(query_type)
    paper_ids = state.get("paper_ids") or []
    paper_id_filter = paper_ids if paper_ids else None
    if paper_id_filter:
        logger.info("Retrieve scope: paper_ids=%s", paper_id_filter)

    def _invoke(q: str) -> list[Document]:
        """同步检索一次（会阻塞，故下面放到线程池里跑）。"""
        docs = retriever.invoke(
            q,
            section_type_filter=section_type_filter,
            paper_id_filter=paper_id_filter,
        )
        if not docs and section_type_filter:
            docs = retriever.invoke(
                q,
                section_type_filter=None,
                paper_id_filter=paper_id_filter,
            )
        return docs

    loop = asyncio.get_event_loop()  # 获取当前事件循环
    # 对每个 query 在线程池里并行调用 _invoke（检索是 CPU/IO 混合，不宜直接阻塞 async）
    results = await asyncio.gather(*[
        loop.run_in_executor(None, _invoke, q) for q in queries
    ])  # results 形如 [[doc1, doc2], [doc3], ...]，每个元素对应一个 query 的检索结果

    trace = get_trace(state.get("trace_id"))
    if trace:
        for q, batch in zip(queries, results):
            trace.retrieval(
                sub_query=query,
                search_queries=[q],
                section_type_filter=section_type_filter,
                paper_id_filter=paper_id_filter,
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

    resp = await llm.ainvoke([
        SystemMessage(content=GENERATOR),  # 系统提示：仅依据上下文、必须引用、学术语气等
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])

    answer = resp.content
    trace = get_trace(state.get("trace_id"))
    if trace:
        trace.generate(
            sub_query=query,
            context_count=len(documents),
            answer=str(answer),
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
        return {"is_sufficient": True, "retry_queries": [], "retries": retries + 1}
    if _is_insufficient_answer(answer):
        return {"is_sufficient": True, "retry_queries": [], "retries": retries + 1}

    # 约束 LLM 输出为 ReflectionResult（is_sufficient + retry_queries）
    structured_llm = llm.with_structured_output(ReflectionResult)
    try:
        result = await structured_llm.ainvoke([
            SystemMessage(content=REFLECTOR),
            HumanMessage(content=f"Question: {query}\nAnswer: {answer}"),
        ])
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
                resp = await llm.ainvoke([
                    SystemMessage(content=GENERATOR),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
                ])
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
