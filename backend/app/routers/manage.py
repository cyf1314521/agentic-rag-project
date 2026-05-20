"""
集合管理与健康检查 API。

- DELETE /api/collection：清空 Milvus 父子集合、上传目录、图表缓存、files 表
- GET /api/health：探测 Milvus 与 LLM 是否可用
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter
from pymilvus import connections, utility

from config import Config
from app.store import clear_all_files
from app.dependencies import get_retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["management"])


@router.delete("/collection")
async def clear_collection():
    """
    危险操作：删除向量库集合并清理本地文件（无鉴权，仅适合开发/内网）。
    """
    try:
        alias = "clear_conn"
        connections.connect(alias=alias, uri=Config.MILVUS_URI)
        dropped = []
        for suffix in ("_children", "_parents"):
            name = f"{Config.COLLECTION_NAME}{suffix}"
            if utility.has_collection(name, using=alias):
                utility.drop_collection(name, using=alias)
                dropped.append(name)
        connections.disconnect(alias)

        # 清空 langchain-milvus 内部 collection 缓存，避免指向已删集合
        retriever = get_retriever()
        for store in (retriever._child_store, retriever._parent_store):
            store._col_cache = None
            store._cache_key = None

        upload_dir = Path(Config.UPLOAD_DIR)
        if upload_dir.exists():
            for f in upload_dir.iterdir():
                if f.suffix == ".pdf":
                    f.unlink(missing_ok=True)

        figures_dir = Path("data/figures")
        if figures_dir.exists():
            shutil.rmtree(figures_dir)

        await clear_all_files()

        return {"ok": True, "dropped": dropped}
    except Exception as e:
        logger.exception("Failed to clear collection")
        return {"ok": False, "detail": str(e)}


@router.get("/health")
async def health():
    """
    健康检查：milvus 能否 list_collections；llm 能否响应简单 invoke。
    """
    status = {"milvus": False, "llm": False}

    try:
        alias = "health_conn"
        connections.connect(alias=alias, uri=Config.MILVUS_URI)
        utility.list_collections(using=alias)
        connections.disconnect(alias)
        status["milvus"] = True
    except Exception:
        pass

    try:
        from app.dependencies import get_llm
        llm = get_llm()
        if llm:
            resp = llm.invoke("Reply with exactly one word: ok")
            status["llm"] = bool((resp.content or "").strip())
    except Exception:
        pass

    ok = all(status.values())
    return {"ok": ok, **status}
