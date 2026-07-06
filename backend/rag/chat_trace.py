"""
聊天链路可观测：RAG 召回与 Agent 各阶段结构化日志 / 可选落盘。

通过 ContextVar + trace_id 注册表关联并行子 Agent；在 chat 入口 start/finish。
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from langchain_core.documents import Document

from config import Config

logger = logging.getLogger(__name__)

_current_trace_id: ContextVar[Optional[str]] = ContextVar("chat_trace_id", default=None)
_registry: dict[str, "ChatTrace"] = {}


def _preview(text: str, limit: Optional[int] = None) -> str:
    n = limit if limit is not None else Config.CHAT_TRACE_PREVIEW_CHARS
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= n:
        return t
    return t[: n - 3] + "..."


def _doc_hit(rank: int, doc: Document) -> dict[str, Any]:
    meta = doc.metadata or {}
    return {
        "rank": rank,
        "paper_id": meta.get("paper_id"),
        "chunk_id": meta.get("chunk_id"),
        "node_type": meta.get("node_type"),
        "section_type": meta.get("section_type"),
        "page_num": meta.get("page_num"),
        "preview": _preview(doc.page_content or ""),
    }


class ChatTrace:
    """单次 /api/chat 请求的追踪记录。"""

    def __init__(self, trace_id: str, session_id: str, query: str) -> None:
        self.trace_id = trace_id
        self.session_id = session_id
        self.query = query
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.events: list[dict[str, Any]] = []

    def _emit(self, stage: str, **payload: Any) -> None:
        if not Config.CHAT_TRACE:
            return
        event = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "stage": stage,
            "at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self.events.append(event)
        logger.info("CHAT_TRACE %s", json.dumps(event, ensure_ascii=False))

    def classify(self, query_type: str) -> None:
        self._emit("classify", query_type=query_type)

    def analyze(self, sub_queries: list[str], *, complexity: str | None = None) -> None:
        payload: dict[str, Any] = {"sub_queries": sub_queries, "count": len(sub_queries)}
        if complexity:
            payload["complexity"] = complexity
        self._emit("analyze", **payload)

    def retrieval(
        self,
        *,
        sub_query: str,
        search_queries: list[str],
        section_type_filter: Optional[list[str]],
        docs: list[Document],
        merged_total: Optional[int] = None,
        paper_id_filter: Optional[list[str]] = None,
    ) -> None:
        self._emit(
            "retrieval",
            sub_query=sub_query,
            search_queries=search_queries,
            section_type_filter=section_type_filter,
            paper_id_filter=paper_id_filter,
            hit_count=len(docs),
            merged_total=merged_total,
            hits=[_doc_hit(i + 1, d) for i, d in enumerate(docs)],
        )

    def generate(
        self,
        *,
        sub_query: str,
        context_count: int,
        answer: str,
        needs_vlm: bool = False,
    ) -> None:
        self._emit(
            "generate",
            sub_query=sub_query,
            context_count=context_count,
            needs_vlm=needs_vlm,
            answer_preview=_preview(str(answer)),
        )

    def reflect(
        self,
        *,
        sub_query: str,
        is_sufficient: bool,
        retry_queries: list[str],
        retries: int,
        answer_preview: str = "",
    ) -> None:
        self._emit(
            "reflect",
            sub_query=sub_query,
            is_sufficient=is_sufficient,
            retry_queries=retry_queries,
            retries=retries,
            answer_preview=_preview(answer_preview),
        )

    def sub_agent_done(
        self,
        *,
        sub_query: str,
        answer: str,
        citation_count: int,
        status: str = "ok",
    ) -> None:
        self._emit(
            "sub_agent_done",
            sub_query=sub_query,
            citation_count=citation_count,
            answer_preview=_preview(str(answer)),
            status=status,
        )

    def sub_agent_failed(self, *, sub_query: str, error: str) -> None:
        self._emit(
            "sub_agent_failed",
            sub_query=sub_query,
            error=error,
        )

    def prepare_synthesis(self, sub_answers: Sequence[Mapping[str, Any]]) -> None:
        items = []
        for sa in sub_answers:
            items.append({
                "query": sa.get("query"),
                "answer_preview": _preview(str(sa.get("answer", ""))),
                "citation_count": len(sa.get("citations") or []),
            })
        self._emit(
            "prepare_synthesis",
            sub_answer_count=len(items),
            total_citations=sum(x["citation_count"] for x in items),
            sub_answers=items,
        )

    def synthesize(self, answer: str) -> None:
        self._emit("synthesize", answer_preview=_preview(str(answer)))

    def error(self, message: str) -> None:
        self._emit("error", error=message)

    def save(self) -> Optional[str]:
        if not Config.CHAT_TRACE_SAVE:
            return None
        out_dir = Path(Config.CHAT_TRACE_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_session = self.session_id.replace(":", "_")[:36]
        path = out_dir / safe_session / f"{self.trace_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "events": self.events,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("CHAT_TRACE saved %s", path)
        return str(path)


def start_trace(session_id: str, query: str, trace_id: Optional[str] = None) -> Optional[ChatTrace]:
    if not Config.CHAT_TRACE:
        return None
    tid = trace_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    trace = ChatTrace(tid, session_id, query)
    _registry[tid] = trace
    _current_trace_id.set(tid)
    trace._emit("start", query=query)
    return trace


def get_trace(trace_id: Optional[str] = None) -> Optional[ChatTrace]:
    if not Config.CHAT_TRACE:
        return None
    tid = trace_id or _current_trace_id.get()
    if tid:
        return _registry.get(tid)
    return None


def finish_trace(trace_id: Optional[str] = None) -> None:
    tid = trace_id or _current_trace_id.get()
    if not tid:
        return
    trace = _registry.pop(tid, None)
    if trace:
        trace.save()
    try:
        _current_trace_id.set(None)
    except LookupError:
        pass
