"""
入库时为每篇论文生成一条压缩画像（paper_profile），供主题/类型检索与定篇。
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import Config
from .models import PaperNode

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_ABSTRACT_HINTS = ("abstract", "摘要", "summary")
_INTRO_HINTS = ("introduction", "引言", "背景")


class PaperProfileMeta(BaseModel):
    """LLM 结构化输出的论文画像字段。"""

    title_guess: str = Field(description="Inferred paper title or topic label")
    domain: str = Field(description="Research domain, e.g. quantum computing, NLP")
    topics: list[str] = Field(description="3-8 topic phrases")
    methods: list[str] = Field(description="Key methods or techniques, 2-6 items")
    task_type: str = Field(description="e.g. experiment, theory, survey, application")
    keywords: list[str] = Field(description="Search keywords, bilingual if useful")
    summary_zh: str = Field(description="80-150 Chinese chars summarizing the paper")
    summary_en: str = Field(default="", description="Optional 1-2 sentence English summary")


PROFILE_PROMPT = """\
You compress an academic PDF into a retrieval-friendly paper profile.
Use ONLY the provided excerpts (abstract, section titles, sample paragraphs).
Output JSON matching the schema. Be specific on domain, topics, and methods.
If the paper is English-only, still fill summary_zh with a concise Chinese summary when possible.
"""


def collect_profile_source(nodes: list[PaperNode], max_chars: int | None = None) -> str:
    """从解析节点抽取摘要、章节标题与少量正文，控制总长度。"""
    limit = max_chars or Config.PAPER_PROFILE_MAX_SOURCE_CHARS
    headers: list[str] = []
    abstract_parts: list[str] = []
    body_parts: list[str] = []

    for node in nodes:
        if node.node_type == "paper_profile":
            continue
        text = (node.text or "").strip()
        if not text:
            continue
        path = " > ".join(node.section_path) if node.section_path else ""
        combined = f"{path}\n{text}" if path else text

        if node.node_type == "section_header":
            headers.append(text)
            continue

        lower = combined.lower()
        if any(h in lower for h in _ABSTRACT_HINTS) and len(abstract_parts) < 3:
            abstract_parts.append(combined[:2000])
            continue
        if any(h in lower for h in _INTRO_HINTS) and len(body_parts) < 2:
            body_parts.append(combined[:1500])
            continue
        if node.node_type == "paragraph" and len(body_parts) < 4:
            body_parts.append(combined[:800])

    sections = []
    if headers:
        sections.append("Section titles:\n" + "\n".join(f"- {h}" for h in headers[:40]))
    if abstract_parts:
        sections.append("Abstract-like excerpts:\n" + "\n\n".join(abstract_parts))
    if body_parts:
        sections.append("Body excerpts:\n" + "\n\n".join(body_parts))

    blob = "\n\n".join(sections).strip()
    if len(blob) > limit:
        blob = blob[:limit] + "\n…"
    return blob


def _format_retrieval_text(paper_id: str, meta: PaperProfileMeta) -> str:
    topics = ", ".join(meta.topics)
    methods = ", ".join(meta.methods)
    keywords = ", ".join(meta.keywords)
    lines = [
        f"Paper: {paper_id}",
        f"Title: {meta.title_guess}",
        f"Domain: {meta.domain}",
        f"Topics: {topics}",
        f"Methods: {methods}",
        f"Task type: {meta.task_type}",
        f"Keywords: {keywords}",
        f"Summary: {meta.summary_zh}",
    ]
    if meta.summary_en.strip():
        lines.append(meta.summary_en.strip())
    return "\n".join(lines)


def _fallback_profile(paper_id: str, source: str) -> PaperProfileMeta:
    snippet = source.replace("\n", " ").strip()[:300]
    return PaperProfileMeta(
        title_guess=paper_id,
        domain="unknown",
        topics=[paper_id.replace("_", " ")],
        methods=[],
        task_type="unknown",
        keywords=[paper_id],
        summary_zh=snippet or f"论文 {paper_id}",
        summary_en="",
    )


def build_paper_profile_node(
    nodes: list[PaperNode],
    paper_id: str,
    llm: BaseChatModel | None = None,
) -> PaperNode | None:
    """生成 paper_profile 节点；失败时退回规则摘要。"""
    source = collect_profile_source(nodes)
    if not source.strip():
        logger.warning("No source text for paper profile: %s", paper_id)
        return None

    meta: PaperProfileMeta
    if llm is not None:
        try:
            structured = llm.with_structured_output(PaperProfileMeta)
            result = structured.invoke([
                SystemMessage(content=PROFILE_PROMPT),
                HumanMessage(content=f"paper_id: {paper_id}\n\n{source}"),
            ])
            meta = PaperProfileMeta.model_validate(result)
        except Exception as exc:
            logger.warning("LLM paper profile failed for %s: %s", paper_id, exc)
            meta = _fallback_profile(paper_id, source)
    else:
        meta = _fallback_profile(paper_id, source)

    retrieval_text = _format_retrieval_text(paper_id, meta)
    return PaperNode(
        node_id=str(uuid.uuid4()),
        paper_id=paper_id,
        node_type="paper_profile",
        text=retrieval_text,
        page_num=0,
        order=-1,
        section_path=["Paper Profile"],
        metadata={
            "section_type": "other",
            "title_guess": meta.title_guess,
            "domain": meta.domain,
            "topics": json.dumps(meta.topics, ensure_ascii=False),
            "methods": json.dumps(meta.methods, ensure_ascii=False),
            "task_type": meta.task_type,
            "keywords": json.dumps(meta.keywords, ensure_ascii=False),
        },
    )


def discover_paper_ids_by_profile(
    query: str,
    session_paper_ids: list[str],
    retriever: Any,
    *,
    top_k: int | None = None,
) -> list[str]:
    """在会话内按 paper_profile 检索，返回按相关度排序的唯一 paper_id。"""
    if not session_paper_ids:
        return []
    k = top_k or Config.PROFILE_DISCOVERY_TOP_K
    try:
        docs = retriever.invoke(
            query,
            paper_id_filter=session_paper_ids,
            node_type_filter=["paper_profile"],
        )
    except TypeError:
        docs = retriever.invoke(query, paper_id_filter=session_paper_ids)
        docs = [d for d in docs if (d.metadata or {}).get("node_type") == "paper_profile"]

    ordered: list[str] = []
    for doc in docs[:k]:
        pid = (doc.metadata or {}).get("paper_id")
        if pid and pid not in ordered:
            ordered.append(str(pid))
    return ordered
