"""
应用依赖与生命周期管理。

在 FastAPI lifespan 内完成「重量级」组件的单例初始化，避免每个请求重复加载模型。
对外提供 get_llm / get_retriever 等访问器，供路由与 Agent 图构建使用。
"""

import sys
import asyncio

# Windows：须在导入 psycopg 之前设置（若仍用 uvicorn 命令行启动，请改用 backend/run.py）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, cast
from pathlib import Path

import httpx
import psycopg
from psycopg_pool import AsyncConnectionPool

from app.db import create_database, execute
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from config import Config
from rag.retrieval import Retriever
from rag.citation import CitationExtractor
from rag.factory import resolve_torch_device
from rag.integration import PDFParser, RAGIntegration
from app.store import init_store

logger = logging.getLogger(__name__)


def _build_llm() -> ChatOpenAI:
    """创建主 LLM；本地 Ollama 时绕过系统 HTTP 代理，避免对 localhost:11434 返回 502。"""
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

    timeout = float(Config.LLM_TIMEOUT)
    llm_kwargs: dict = {
        "base_url": Config.LLM_BASE_URL,
        "model": Config.LLM_MODEL,
        "api_key": Config.LLM_API_KEY or "ollama",
        "temperature": Config.LLM_TEMPERATURE,
        "max_tokens": Config.LLM_MAX_TOKENS,
        "timeout": timeout,
        "max_retries": 1,
    }
    base = (Config.LLM_BASE_URL or "").lower()
    # 127.0.0.1 比 localhost 更不易被系统代理劫持
    if "11434" in base or "127.0.0.1" in base or "localhost" in base:
        llm_kwargs["extra_body"] = {"think": False}
        llm_kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
        llm_kwargs["http_async_client"] = httpx.AsyncClient(trust_env=False, timeout=timeout)
    return ChatOpenAI(**llm_kwargs)


def _build_vlm_llm() -> ChatOpenAI:
    """创建 VLM（图表理解）；配置与 _build_llm 相同，使用 VLM_* 环境变量。"""
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

    timeout = float(Config.LLM_TIMEOUT)
    llm_kwargs: dict = {
        "base_url": Config.VLM_BASE_URL,
        "model": Config.VLM_MODEL,
        "api_key": Config.VLM_API_KEY or "ollama",
        "temperature": 0,
        "timeout": timeout,
        "max_retries": 1,
    }
    base = (Config.VLM_BASE_URL or "").lower()
    if "11434" in base or "127.0.0.1" in base or "localhost" in base:
        llm_kwargs["extra_body"] = {"think": False}
        llm_kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
        llm_kwargs["http_async_client"] = httpx.AsyncClient(trust_env=False, timeout=timeout)
    return ChatOpenAI(**llm_kwargs)


async def _ensure_postgres_db():
    """
    若目标数据库不存在则自动创建。

    连接到默认库 postgres，查询 datname 后执行 CREATE DATABASE。
    失败时仅打警告，不阻断启动（可能库已存在或权限不足）。
    """
    uri = Config.POSTGRES_URI
    idx = uri.rfind("/")
    db_name = uri[idx + 1:]
    base_uri = uri[:idx] + "/postgres"
    try:
        conn = await psycopg.AsyncConnection.connect(base_uri, autocommit=True)
        async with conn:
            cur = await execute(conn, "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if not await cur.fetchone():
                await create_database(conn, db_name)
                logger.info(f"Created database: {db_name}")
    except Exception as e:
        logger.warning(f"Could not auto-create database: {e}")


class RetrieverTool:
    """
    供 LangGraph 子 Agent 调用的检索适配器。

    封装 Retriever.retrieve 的默认参数（TOP_K、RRF、重排、父块扩展等）。
    """

    def __init__(self, retriever: Retriever):
        self._retriever = retriever

    def invoke(self, query: str, section_type_filter=None, paper_id_filter=None):
        """执行一次完整检索管道，返回 Document 列表。"""
        return self._retriever.retrieve(
            query=query,
            k=Config.TOP_K,
            use_hyde=False,
            rerank=True,
            expand_parent=True,
            rrf_k=Config.RRF_K,
            fetch_k=Config.FETCH_K,
            section_type_filter=section_type_filter,
            paper_id_filter=paper_id_filter,
        )


# ---------- 进程级单例（lifespan 内赋值）----------
_llm: ChatOpenAI | None = None
_retriever: Retriever | None = None
_retriever_tool: RetrieverTool | None = None
_pdf_parser: PDFParser | None = None
_rag_integration: RAGIntegration | None = None
_checkpointer: AsyncPostgresSaver | None = None


def get_llm() -> ChatOpenAI:
    """获取已初始化的主 LLM 客户端。"""
    return _llm  # type: ignore


def get_retriever() -> Retriever:
    """获取混合检索器（文件上传去重、删除向量等场景使用）。"""
    return _retriever  # type: ignore


def get_retriever_tool() -> RetrieverTool:
    """获取供 Agent 图使用的检索工具包装。"""
    return _retriever_tool  # type: ignore


def ensure_retriever_tool() -> RetrieverTool:
    """
    CLI 脚本在 FastAPI lifespan 外运行时初始化 Retriever（懒加载单例）。
    run_scope_eval.py、debug 脚本等应使用本函数而非 get_retriever_tool()。
    """
    global _retriever, _retriever_tool
    if _retriever_tool is None:
        logger.info("Bootstrapping Retriever for standalone script …")
        _retriever = Retriever(
            embedding_model=Config.EMBEDDING_MODEL,
            reranker_model=Config.RERANKER_MODEL,
            milvus_uri=Config.MILVUS_URI,
            collection_name=Config.COLLECTION_NAME,
            enable_cache=Config.ENABLE_CACHE,
        )
        _retriever_tool = RetrieverTool(_retriever)
    return _retriever_tool


def get_pdf_parser() -> PDFParser:
    """获取 PDF 解析器（含 Docling + 可选 LLM 章节分类）。"""
    return _pdf_parser  # type: ignore


def get_rag_integration() -> RAGIntegration:
    """获取节点转 Document、父子分块等 RAG 集成逻辑。"""
    return _rag_integration  # type: ignore


def get_checkpointer() -> AsyncPostgresSaver:
    """获取 LangGraph 异步 Postgres 检查点存储（多轮对话状态）。"""
    return _checkpointer  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期钩子。

    启动：建库 → 连接池 → 初始化 store 表 → 加载 LLM/Retriever/Parser → 设置 checkpointer
    关闭：释放 checkpointer 与连接池
    """
    global _llm, _retriever, _retriever_tool, _pdf_parser, _rag_integration, _checkpointer

    logger.info("Starting up — loading models …")

    if sys.platform == "win32" and type(asyncio.get_running_loop()).__name__ == "ProactorEventLoop":
        raise RuntimeError(
            "Windows 上 psycopg 异步需要 SelectorEventLoop。"
            "请勿使用 `python -m uvicorn`，请改用: .\\.venv\\Scripts\\python.exe run.py"
        )

    await _ensure_postgres_db()

    pool = AsyncConnectionPool(Config.POSTGRES_URI, min_size=2, max_size=10, open=False)
    await pool.open()
    await init_store(cast(Any, pool))

    _llm = _build_llm()

    embed_device = resolve_torch_device()
    logger.info("Embedding/reranker device: %s (EMBEDDING_DEVICE=%s)", embed_device, Config.EMBEDDING_DEVICE)

    _retriever = Retriever(
        embedding_model=Config.EMBEDDING_MODEL,
        reranker_model=Config.RERANKER_MODEL,
        milvus_uri=Config.MILVUS_URI,
        collection_name=Config.COLLECTION_NAME,
        enable_cache=Config.ENABLE_CACHE,
    )
    _retriever_tool = RetrieverTool(_retriever)

    _pdf_parser = PDFParser(llm=_llm)
    _rag_integration = RAGIntegration(
        embedding_model=Config.EMBEDDING_MODEL,
        milvus_uri=Config.MILVUS_URI,
        collection_name=Config.COLLECTION_NAME,
    )

    upload_dir = Path(Config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # checkpointer 在 yield 期间保持打开，供请求内读写 thread 状态
    async with AsyncPostgresSaver.from_conn_string(Config.POSTGRES_URI) as cp:
        await cp.setup()
        _checkpointer = cp
        logger.info("Startup complete.")
        yield

    _checkpointer = None
    await pool.close()
    logger.info("Shutting down.")
