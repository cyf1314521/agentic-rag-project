"""
会话语料表征：为 Corpus Gate 构建每篇论文的 gate_text / gate_vector（不查 Milvus）。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from config import Config
from rag.parse_artifact import load_paper_nodes_from_artifact
from rag.paper_profile import (
    _fallback_profile,
    _format_retrieval_text,
    collect_profile_source,
)

logger = logging.getLogger(__name__)

GateSource = Literal["profile", "parse_fallback", "filename_only"]

_CORPUS_CACHE: dict[str, "SessionCorpus"] = {}


@dataclass
class PaperGateRepr:
    paper_id: str
    gate_text: str
    gate_vector: list[float] | None = None
    source: GateSource = "parse_fallback"
    profile_status: str = "unknown"


@dataclass
class SessionCorpus:
    paper_ids: list[str]
    reprs: list[PaperGateRepr] = field(default_factory=list)
    corpus_id: str = ""

    def by_paper_id(self) -> dict[str, PaperGateRepr]:
        return {r.paper_id: r for r in self.reprs}


def corpus_cache_key(paper_ids: list[str], sources: dict[str, str]) -> str:
    payload = "|".join(sorted(paper_ids)) + "|" + "|".join(
        f"{pid}:{sources.get(pid, '')}" for pid in sorted(paper_ids)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def invalidate_corpus_cache() -> None:
    _CORPUS_CACHE.clear()


def _artifact_path(paper_id: str) -> Path | None:
    base = Path(Config.PARSE_ARTIFACT_DIR) / paper_id
    if not base.is_dir():
        return None
    jsons = sorted(base.glob("*.json"))
    return jsons[0] if jsons else None


def _gate_text_from_parse_artifact(paper_id: str) -> tuple[str, GateSource]:
    path = _artifact_path(paper_id)
    if not path:
        return f"Paper: {paper_id}\nTitle: {paper_id}", "filename_only"
    try:
        nodes = load_paper_nodes_from_artifact(path)
        source = collect_profile_source(nodes, max_chars=Config.PAPER_PROFILE_MAX_SOURCE_CHARS)
        if not source.strip():
            data = json.loads(path.read_text(encoding="utf-8"))
            filename = data.get("meta", {}).get("filename", paper_id)
            return f"Paper: {paper_id}\nFilename: {filename}", "filename_only"
        meta = _fallback_profile(paper_id, source)
        return _format_retrieval_text(paper_id, meta), "parse_fallback"
    except Exception as exc:
        logger.warning("gate_text from artifact failed for %s: %s", paper_id, exc)
        return f"Paper: {paper_id}", "filename_only"


def build_paper_gate_repr(
    paper_id: str,
    *,
    profile_status: str = "unknown",
    gate_text: str | None = None,
    source: GateSource | None = None,
) -> PaperGateRepr:
    if gate_text is None:
        gate_text, src = _gate_text_from_parse_artifact(paper_id)
        source = source or src
    return PaperGateRepr(
        paper_id=paper_id,
        gate_text=gate_text,
        source=source or "parse_fallback",
        profile_status=profile_status,
    )


def load_session_corpus(
    paper_ids: list[str],
    *,
    profile_statuses: dict[str, str] | None = None,
) -> SessionCorpus:
    """同步加载会话语料（parse artifact 为主；profile 向量在 Milvus，此处用摘要文本）。"""
    statuses = profile_statuses or {}
    sources = {pid: statuses.get(pid, "unknown") for pid in paper_ids}
    key = corpus_cache_key(paper_ids, sources)
    if key in _CORPUS_CACHE:
        return _CORPUS_CACHE[key]

    reprs = [
        build_paper_gate_repr(pid, profile_status=statuses.get(pid, "unknown"))
        for pid in paper_ids
    ]
    corpus = SessionCorpus(paper_ids=list(paper_ids), reprs=reprs, corpus_id=key)
    _CORPUS_CACHE[key] = corpus
    return corpus


def attach_gate_vectors(corpus: SessionCorpus, embeddings: Any) -> SessionCorpus:
    """为 corpus 内各 paper 计算并附加 gate_vector（内存缓存，不写入 Milvus）。"""
    texts = [r.gate_text for r in corpus.reprs]
    try:
        vectors = embeddings.embed_documents(texts)
    except Exception as exc:
        logger.warning("embed_documents for gate failed: %s", exc)
        vectors = []
    for repr_, vec in zip(corpus.reprs, vectors):
        repr_.gate_vector = list(vec) if vec is not None else None
    return corpus
