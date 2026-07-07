"""
PDF 上传与文件管理 API。

入库流程：校验 → SHA256 去重 → 落盘 → Docling 解析 → 父子分块 → 写入 Milvus → 记录 files 表
paper_profile 在核心入库成功后由 BackgroundTasks 异步追加（不阻塞 HTTP）。
"""

import uuid
import hashlib
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
import aiofiles

from config import Config
from app.dependencies import get_pdf_parser, get_rag_integration, get_retriever
from app.ingest_tasks import build_paper_profile_background
from app.store import (
    add_file,
    list_files,
    delete_file_record,
    get_file,
    create_session,
    link_session_paper,
    update_file_profile_status,
)
from rag.parse_artifact import ParseRecorder, save_parse_artifact, delete_parse_artifact

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/files", tags=["files"])


@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(None),
):
    """
    POST /api/files/upload — 批量上传 PDF（multipart）。

    单文件可能返回 status: ok | duplicate | error
    核心正文入库成功后立即返回；paper_profile 异步生成（profile_status=pending）。
    """
    upload_dir = Path(Config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    results = []

    if not session_id:
        session_id = str(uuid.uuid4())
    await create_session(session_id)

    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            results.append({"filename": f.filename, "status": "error", "detail": "Only PDF files are supported"})
            continue

        content = await f.read()
        if len(content) > max_bytes:
            results.append({"filename": f.filename, "status": "error", "detail": f"File exceeds {Config.MAX_UPLOAD_SIZE_MB}MB limit"})
            continue

        content_hash = hashlib.sha256(content).hexdigest()

        retriever = get_retriever()
        updater = retriever.get_updater()
        existing_paper = updater.has_content_hash(content_hash)
        if existing_paper:
            if session_id:
                await link_session_paper(session_id, existing_paper)
            results.append({
                "filename": f.filename,
                "status": "duplicate",
                "detail": (
                    f"内容与已入库论文 '{existing_paper}' 相同，未重复解析；"
                    f"已绑定到当前会话，可直接提问。"
                ),
                "paper_id": existing_paper,
                "linked_to_session": bool(session_id),
            })
            continue

        file_id = str(uuid.uuid4())
        paper_id = Path(f.filename).stem
        save_path = upload_dir / f"{file_id}.pdf"

        async with aiofiles.open(save_path, "wb") as out:
            await out.write(content)

        recorder = None
        if Config.SAVE_PARSE_ARTIFACT:
            recorder = ParseRecorder(
                file_id=file_id,
                paper_id=paper_id,
                filename=f.filename,
                pdf_path=str(save_path),
                content_hash=content_hash,
            )

        artifact_path: Path | None = None
        try:
            parser = get_pdf_parser()
            nodes = parser.parse(str(save_path), paper_id, recorder=recorder)

            integration = get_rag_integration()
            docs = integration.nodes_to_documents(nodes, content_hash=content_hash)
            parents, children = integration.create_chunks(docs)

            if recorder:
                recorder.stage(
                    "chunking",
                    {
                        "parent_chunks": len(parents),
                        "child_chunks": len(children),
                        "split_types": sorted(
                            {t for d in docs if (t := d.metadata.get("node_type"))}
                        ),
                    },
                )

            updater._ensure_connections()
            updater.parent_store.add_documents(parents)
            updater.child_store.add_documents(children)

            if recorder:
                recorder.set_indexing({
                    "milvus_uri": Config.MILVUS_URI,
                    "collections": [
                        f"{Config.COLLECTION_NAME}_parents",
                        f"{Config.COLLECTION_NAME}_children",
                    ],
                    "parent_chunks": len(parents),
                    "child_chunks": len(children),
                })
                artifact_path = save_parse_artifact(recorder, nodes)
                logger.info("Parse artifact saved: %s", artifact_path)

            profile_status = "pending" if Config.PAPER_PROFILE_ENABLED else "skipped"
            record = await add_file(
                file_id=file_id,
                filename=f.filename,
                paper_id=paper_id,
                size_bytes=len(content),
                page_count=max((n.page_num for n in nodes), default=0),
                chunk_count=len(children),
                profile_status=profile_status,
            )
            if session_id:
                await link_session_paper(session_id, paper_id)

            payload = {"filename": f.filename, "status": "ok", **record}
            if artifact_path:
                payload["parse_artifact"] = str(artifact_path)

            if Config.PAPER_PROFILE_ENABLED and artifact_path:
                background_tasks.add_task(
                    build_paper_profile_background,
                    file_id,
                    paper_id,
                    content_hash,
                    str(artifact_path),
                )
                payload["profile_status"] = "pending"
                payload["detail"] = "正文已入库，论文画像后台生成中（discovery 检索稍后可用）。"
            elif Config.PAPER_PROFILE_ENABLED:
                await update_file_profile_status(file_id, "skipped", profile_error="no parse artifact")
                payload["profile_status"] = "skipped"

            results.append(payload)
        except Exception as e:
            logger.exception(f"Failed to process {f.filename}")
            save_path.unlink(missing_ok=True)
            if recorder and Config.SAVE_PARSE_ARTIFACT:
                delete_parse_artifact(paper_id, file_id)
            results.append({"filename": f.filename, "status": "error", "detail": str(e)})

    return {"session_id": session_id, "files": results}


@router.get("")
async def get_files():
    """GET /api/files — 已上传文件列表。"""
    return await list_files()


@router.get("/{file_id}/parse-artifact")
async def get_parse_artifact(file_id: str):
    """GET /api/files/{file_id}/parse-artifact — 返回该次上传保存的解析 JSON。"""
    import json

    record = await get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    path = Path(Config.PARSE_ARTIFACT_DIR) / record["paper_id"] / f"{file_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Parse artifact not found")

    return json.loads(path.read_text(encoding="utf-8"))


@router.delete("/{file_id}")
async def remove_file(file_id: str):
    """
    DELETE /api/files/{file_id} — 删除向量、磁盘 PDF、图表目录及 DB 记录。
    """
    record = await delete_file_record(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    retriever = get_retriever()
    updater = retriever.get_updater()
    updater.delete_paper(record["paper_id"])

    save_path = Path(Config.UPLOAD_DIR) / f"{file_id}.pdf"
    save_path.unlink(missing_ok=True)

    figures_dir = Path("data/figures") / record["paper_id"]
    if figures_dir.exists():
        shutil.rmtree(figures_dir)

    if Config.SAVE_PARSE_ARTIFACT:
        delete_parse_artifact(record["paper_id"], file_id)

    return {"ok": True, "paper_id": record["paper_id"]}
