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

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document

from .states import AgentState, SubAgentState
from .prompts import QUERY_ANALYZER, QUERY_CLASSIFIER, SYNTHESIZER, GENERATOR, REFLECTOR, SUMMARIZER

logger = logging.getLogger(__name__)

# 送入本地 LLM 的检索上下文上限，避免 Ollama 因 prompt 过长返回 502
MAX_GENERATE_CONTEXT_CHARS = 12_000

# 保留最近 N 条消息原文；更早的由 summarize 压缩进 summary
WINDOW_SIZE = 6


class QueryAnalysis(BaseModel):
    """analyze_query 的结构化输出：子查询列表。"""
    sub_queries: list[str] = Field(description="List of sub-queries, original query first")


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


def _remap_citations(answer: str, offset: int) -> str:
    """将答案中的 [1][2] 引用编号整体加上 offset，避免多子答案合并后编号冲突。"""
    def _replace(m):
        return f"[{int(m.group(1)) + offset}]"
    return re.sub(r"\[(\d+)\]", _replace, answer)


# --------------- 主图节点 ---------------

async def analyze_query(state: AgentState, llm: BaseChatModel) -> dict:
    """
    查询分解：将复杂问题拆成多个可独立检索的子查询。
    失败时退化为仅使用原始 query。
    """
    query = state["query"]
    context_header = _build_context_header(state)

    system_content = QUERY_ANALYZER
    if context_header:
        system_content += f"\n\n# Conversation Context\n{context_header}"
    msgs = [SystemMessage(content=system_content), HumanMessage(content=query)]

    structured_llm = llm.with_structured_output(QueryAnalysis)
    try:
        result: QueryAnalysis = await structured_llm.ainvoke(msgs)
        sub_queries = result.sub_queries
    except Exception:
        sub_queries = [query]

    if query not in sub_queries:
        sub_queries.insert(0, query)

    logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
    return {"sub_queries": sub_queries}


async def classify_query(state: AgentState, llm: BaseChatModel) -> dict:
    """
    查询分类：决定检索时是否按 section_type（experiment/method/background）过滤。
    同时写入 ContextVar 供 tools.paper_retrieval 使用。
    """
    from .tools import set_query_type
    query = state["query"]

    structured_llm = llm.with_structured_output(QueryClassification)
    try:
        result: QueryClassification = await structured_llm.ainvoke([
            SystemMessage(content=QUERY_CLASSIFIER),
            HumanMessage(content=query),
        ])
        query_type = result.query_type
        if query_type not in ("experimental_result", "method", "background", "general"):
            query_type = "general"
    except Exception:
        query_type = "general"

    set_query_type(query_type)
    logger.info(f"Query classified as: {query_type}")
    return {"query_type": query_type}


def prepare_synthesis(state: AgentState) -> dict:
    """
    汇总所有 sub_answers，重映射引用编号，构建 synthesize 节点的消息列表。
    """
    sub_answers = state.get("sub_answers", [])
    context_header = _build_context_header(state)

    context_parts = []
    all_citations = []
    global_idx = 0
    for sa in sub_answers:
        remapped = _remap_citations(sa["answer"], global_idx)
        context_parts.append(f"Q: {sa['query']}\nA: {remapped}")
        sa_citations = sa.get("citations", [])
        all_citations.extend(sa_citations)
        global_idx += max(len(sa_citations), 1)

    sub_context = "\n\n".join(context_parts)

    system_content = SYNTHESIZER.format(context=sub_context)
    if context_header:
        system_content += f"\n\n# Conversation Context\n{context_header}"

    return {
        "synth_messages": [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Original question: {state['query']}"),
        ],
        "citations": all_citations,
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
    return {"answer": "".join(chunks)}


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
        "messages": [RemoveMessage(id=m.id) for m in to_summarize],
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
    query = state["query"]
    # 若上一轮 reflect 认为答案不足，会填入补充检索词；首次检索时为空列表
    retry_queries = state.get("retry_queries", [])
    # 主图 classify 节点写入：experimental_result / method / background / general
    query_type = state.get("query_type", "general")

    # 有补充 query 就用它们一起搜；否则只搜当前 query
    queries = retry_queries if retry_queries else [query]

    # 查询类型 → Milvus 元数据字段 section_type 的过滤值（入库时 PDF 解析已打好标签）
    _ROUTE_CONFIG: dict[str, list[str] | None] = {
        "experimental_result": ["experiment"],   # 问实验结果 → 优先搜实验章节
        "method": ["method"],                   # 问方法 → 优先搜方法章节
        "background": ["background"],             # 问背景/动机 → 优先搜引言等
        "general": None,                        # 通用问题 → 不过滤章节
    }
    # 根据 query_type 取出过滤条件；没有则 None 表示全库检索
    section_type_filter = _ROUTE_CONFIG.get(query_type)

    def _invoke(q: str) -> list[Document]:
        """同步检索一次（会阻塞，故下面放到线程池里跑）。"""
        # retriever.invoke → Retriever.retrieve：BM25+向量、重排、父块扩展
        docs = retriever.invoke(q, section_type_filter=section_type_filter)
        # 若「只在实验章」里没搜到，再放开过滤搜全文，避免漏答
        if not docs and section_type_filter:
            docs = retriever.invoke(q, section_type_filter=None)
        return docs  # list[Document]，每条含 page_content 与 metadata

    loop = asyncio.get_event_loop()  # 获取当前事件循环
    # 对每个 query 在线程池里并行调用 _invoke（检索是 CPU/IO 混合，不宜直接阻塞 async）
    results = await asyncio.gather(*[
        loop.run_in_executor(None, _invoke, q) for q in queries
    ])  # results 形如 [[doc1, doc2], [doc3], ...]，每个元素对应一个 query 的检索结果

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

    return {"answer": resp.content}  # 写入 state["answer"]，供 reflect 评估


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
        # 没检索到内容时无法反思，直接视为「结束」，避免死循环
        return {"is_sufficient": True, "retry_queries": [], "retries": retries + 1}

    # 约束 LLM 输出为 ReflectionResult（is_sufficient + retry_queries）
    structured_llm = llm.with_structured_output(ReflectionResult)
    try:
        result: ReflectionResult = await structured_llm.ainvoke([
            SystemMessage(content=REFLECTOR),
            HumanMessage(content=f"Question: {query}\nAnswer: {answer}"),
        ])
        is_sufficient = result.is_sufficient       # True/False
        retry_queries = result.retry_queries       # 例如缺 latency 时 ["What is the latency of method X?"]
    except Exception:
        # 结构化输出失败时保守处理：不再重试，防止卡死
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
                return {
                    "answer": resp.content,           # 覆盖原 answer
                    "documents": enhanced_docs,       # 更新上下文
                    "needs_vlm": True,                # 标记已用 VLM，防止再次兜底
                    "is_sufficient": True,            # 强制结束反思循环
                    "retry_queries": [],              # 不再走文本重检索
                    "retries": retries + 1,
                }

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
