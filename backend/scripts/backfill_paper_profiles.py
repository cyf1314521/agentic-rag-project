"""
为已入库 PDF 补写 paper_profile chunk（不重跑 Docling）。

用法（backend 目录）:
  python scripts/backfill_paper_profiles.py
  python scripts/backfill_paper_profiles.py --paper-id eval_05_wind_turbine
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

from config import Config
from app.dependencies import _build_llm, ensure_retriever_tool, get_rag_integration
from rag.integration import RAGIntegration
from rag.paper_profile import build_paper_profile_node
from rag.models import PaperNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_nodes_from_artifact(paper_id: str) -> list[PaperNode]:
    """从 parse artifact 重建轻量 nodes（仅用于 profile 源文本）。"""
    artifact_dir = Path(Config.PARSE_ARTIFACT_DIR) / paper_id
    if not artifact_dir.is_dir():
        return []
    nodes: list[PaperNode] = []
    import json

    for path in sorted(artifact_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data.get("nodes") or data.get("stages", [{}])[-1].get("nodes") or []:
            if isinstance(item, dict) and item.get("text"):
                nodes.append(
                    PaperNode(
                        node_id=item.get("node_id", "x"),
                        paper_id=paper_id,
                        node_type=item.get("node_type", "paragraph"),
                        text=item["text"],
                        page_num=int(item.get("page_num", 1)),
                        order=int(item.get("order", 0)),
                        section_path=item.get("section_path") or [],
                        metadata=item.get("metadata") or {},
                    )
                )
    return nodes


def backfill(paper_ids: list[str], *, dry_run: bool = False) -> None:
    llm = _build_llm()
    integration = get_rag_integration() or RAGIntegration(
        embedding_model=Config.EMBEDDING_MODEL,
        milvus_uri=Config.MILVUS_URI,
        collection_name=Config.COLLECTION_NAME,
    )
    retriever = ensure_retriever_tool()
    updater = retriever._retriever.get_updater()
    updater._ensure_connections()

    for paper_id in paper_ids:
        nodes = _load_nodes_from_artifact(paper_id)
        if not nodes:
            logger.warning("Skip %s: no parse artifact nodes", paper_id)
            continue
        profile = build_paper_profile_node(nodes, paper_id, llm)
        if not profile:
            logger.warning("Skip %s: profile build failed", paper_id)
            continue
        if dry_run:
            logger.info("Would index profile for %s (%d chars)", paper_id, len(profile.text))
            continue
        docs = integration.nodes_to_documents([profile], content_hash="profile_backfill")
        parents, children = integration.create_chunks(docs)
        updater.parent_store.add_documents(parents)
        updater.child_store.add_documents(children)
        logger.info("Indexed paper_profile for %s", paper_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.paper_id:
        ids = args.paper_id
    else:
        from eval.retrieval_eval_runner import all_eval_paper_ids
        import json

        dataset = _BACKEND / "eval" / "fixtures" / "scope_eval_dataset.json"
        cases = json.loads(dataset.read_text(encoding="utf-8"))["cases"]
        ids = all_eval_paper_ids(cases)

    backfill(ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
