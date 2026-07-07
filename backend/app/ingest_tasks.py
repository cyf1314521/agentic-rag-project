"""
入库后台任务：paper_profile 与核心解析解耦，避免上传 HTTP 被 LLM 阻塞。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from config import Config
from app.dependencies import get_llm, get_rag_integration, get_retriever
from app.store import update_file_profile_status
from rag.parse_artifact import load_paper_nodes_from_artifact
from rag.paper_profile import build_paper_profile_node
from agent.session_corpus import invalidate_corpus_cache

logger = logging.getLogger(__name__)


def _build_and_index_profile_sync(
    file_id: str,
    paper_id: str,
    content_hash: str,
    artifact_path: str,
) -> tuple[str, int]:
    """同步：从 artifact 重建节点 → LLM 画像 → 追加 Milvus。"""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"parse artifact missing: {artifact_path}")

    nodes = load_paper_nodes_from_artifact(path)
    if not nodes:
        return "skipped", 0

    profile_node = build_paper_profile_node(nodes, paper_id, get_llm())
    if profile_node is None:
        return "skipped", 0

    integration = get_rag_integration()
    docs = integration.nodes_to_documents([profile_node], content_hash=content_hash)
    parents, children = integration.create_chunks(docs)

    updater = get_retriever().get_updater()
    updater.append_documents(parents, children)
    return "ready", len(children)


async def build_paper_profile_background(
    file_id: str,
    paper_id: str,
    content_hash: str,
    artifact_path: str,
) -> None:
    """FastAPI BackgroundTasks 入口：画像生成失败不影响已入库正文。"""
    if not Config.PAPER_PROFILE_ENABLED:
        await update_file_profile_status(file_id, "skipped")
        return

    await update_file_profile_status(file_id, "processing")
    loop = asyncio.get_running_loop()
    try:
        status, chunk_count = await loop.run_in_executor(
            None,
            _build_and_index_profile_sync,
            file_id,
            paper_id,
            content_hash,
            artifact_path,
        )
        await update_file_profile_status(
            file_id,
            status,
            profile_chunk_count=chunk_count,
        )
        invalidate_corpus_cache()
        logger.info(
            "Background paper_profile %s for %s (%d chunks)",
            status,
            paper_id,
            chunk_count,
        )
    except Exception as exc:
        logger.exception("Background paper_profile failed for %s: %s", paper_id, exc)
        await update_file_profile_status(file_id, "failed", profile_error=str(exc)[:500])
