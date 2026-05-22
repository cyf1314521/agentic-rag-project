"""
PDF 解析可观测性：记录解析阶段、统计与节点全文，每次上传写入单个 JSON 文件。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import Config
from .models import PaperNode

SCHEMA_VERSION = 1


class ParseRecorder:
    """在 PDFParser 与上传流程中累积解析过程信息。"""

    def __init__(
        self,
        *,
        file_id: str,
        paper_id: str,
        filename: str,
        pdf_path: str,
        content_hash: str = "",
    ):
        self.file_id = file_id
        self.paper_id = paper_id
        self.filename = filename
        self.pdf_path = pdf_path
        self.content_hash = content_hash
        self._started = time.perf_counter()
        self._stages: list[dict[str, Any]] = []
        self._warnings: list[str] = []
        self._statistics: dict[str, Any] = {}
        self._section_classification: dict[str, Any] = {"status": "pending"}
        self._indexing: dict[str, Any] = {}

    def stage(self, name: str, detail: Optional[dict[str, Any]] = None, *, duration_sec: Optional[float] = None):
        entry: dict[str, Any] = {
            "name": name,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        if detail:
            entry["detail"] = detail
        self._stages.append(entry)

    def warn(self, message: str):
        self._warnings.append(message)

    def set_statistics(self, stats: dict[str, Any]):
        self._statistics.update(stats)

    def set_section_classification(self, status: str, detail: Optional[dict[str, Any]] = None):
        payload: dict[str, Any] = {"status": status}
        if detail:
            payload["detail"] = detail
        self._section_classification = payload

    def set_indexing(self, indexing: dict[str, Any]):
        self._indexing = indexing

    def build_document(self, nodes: list[PaperNode]) -> dict[str, Any]:
        """组装最终写入磁盘的 JSON 文档。"""
        by_type: dict[str, int] = {}
        node_exports: list[dict[str, Any]] = []
        for n in nodes:
            by_type[n.node_type] = by_type.get(n.node_type, 0) + 1
            meta = {k: v for k, v in n.metadata.items() if k != "item"}
            node_exports.append({
                "node_id": n.node_id,
                "order": n.order,
                "node_type": n.node_type,
                "page_num": n.page_num,
                "section_path": n.section_path,
                "section_type": meta.get("section_type", "other"),
                "bbox": list(n.bbox) if n.bbox else None,
                "image_path": n.image_path,
                "text_length": len(n.text),
                "text": n.text,
                "related_ids": n.related_ids,
                "metadata": meta,
            })

        stats = dict(self._statistics)
        stats.setdefault("nodes_total", len(nodes))
        stats.setdefault("by_node_type", by_type)
        stats.setdefault("text_chars_total", sum(len(n.text) for n in nodes))
        stats.setdefault("max_page_num", max((n.page_num for n in nodes), default=0))

        return {
            "schema_version": SCHEMA_VERSION,
            "meta": {
                "file_id": self.file_id,
                "paper_id": self.paper_id,
                "filename": self.filename,
                "pdf_path": self.pdf_path,
                "content_hash": self.content_hash,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "total_duration_sec": round(time.perf_counter() - self._started, 3),
                "config": {
                    "docling_low_memory": Config.DOCLING_LOW_MEMORY,
                    "max_upload_size_mb": Config.MAX_UPLOAD_SIZE_MB,
                    "llm_model": Config.LLM_MODEL,
                    "embedding_model": Config.EMBEDDING_MODEL,
                },
            },
            "stages": self._stages,
            "warnings": self._warnings,
            "statistics": stats,
            "section_classification": self._section_classification,
            "indexing": self._indexing,
            "nodes": node_exports,
        }


def save_parse_artifact(recorder: ParseRecorder, nodes: list[PaperNode]) -> Path:
    """写入 data/parsed/{paper_id}/{file_id}.json，返回路径。"""
    base = Path(Config.PARSE_ARTIFACT_DIR)
    out_dir = base / recorder.paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{recorder.file_id}.json"
    document = recorder.build_document(nodes)
    out_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def delete_parse_artifact(paper_id: str, file_id: Optional[str] = None) -> None:
    """删除论文解析记录目录或单个 file 的 JSON。"""
    base = Path(Config.PARSE_ARTIFACT_DIR) / paper_id
    if file_id:
        path = base / f"{file_id}.json"
        path.unlink(missing_ok=True)
        if base.exists() and not any(base.iterdir()):
            base.rmdir()
    elif base.exists():
        import shutil
        shutil.rmtree(base, ignore_errors=True)
