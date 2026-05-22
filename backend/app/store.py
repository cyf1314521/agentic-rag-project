"""
会话与上传文件的元数据存储（PostgreSQL）。

说明：
- 对话消息正文由 LangGraph AsyncPostgresSaver 持久化（checkpoint）
- 本模块仅维护 sessions / files 两张业务表，供侧边栏列表、文件管理使用
"""

import time
from typing import Any, Optional

from psycopg_pool import AsyncConnectionPool

from app.db import cursor_columns, execute

# 全局连接池引用，由 init_store 在 lifespan 中注入
_pool: Optional[AsyncConnectionPool] = None

# 会话表：标题、创建/更新时间
_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
)"""

# 已上传 PDF 的元信息（与 Milvus 中 paper_id 对应）
_FILES_DDL = """
CREATE TABLE IF NOT EXISTS files (
    file_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    size_bytes BIGINT NOT NULL DEFAULT 0,
    page_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL
)"""

# 会话 ↔ 文档范围：该 session 聊天仅检索关联的 paper_id
_SESSION_PAPERS_DDL = """
CREATE TABLE IF NOT EXISTS session_papers (
    session_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    linked_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (session_id, paper_id)
)"""


async def init_store(pool: AsyncConnectionPool):  # type: ignore[type-arg]
    """应用启动时建表（若不存在）。"""
    global _pool
    _pool = pool
    async with pool.connection() as conn:
        await execute(conn, _SESSIONS_DDL)
        await execute(conn, _FILES_DDL)
        await execute(conn, _SESSION_PAPERS_DDL)
        await conn.commit()


def _get_pool() -> AsyncConnectionPool:
    assert _pool is not None, "store not initialised — call init_store first"
    return _pool


# ── 会话 CRUD ─────────────────────────────────────────────

async def create_session(session_id: str, title: str = "") -> dict:
    """创建新会话；若 session_id 已存在则忽略（ON CONFLICT DO NOTHING）。"""
    now = time.time()
    async with _get_pool().connection() as conn:
        await execute(
            conn,
            "INSERT INTO sessions (session_id, title, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (session_id) DO NOTHING",
            (session_id, title, now, now),
        )
        await conn.commit()
    return {"session_id": session_id, "title": title, "created_at": now, "updated_at": now}


async def update_session(session_id: str, title: Optional[str] = None) -> bool:
    """更新会话；可只更新 title 或刷新 updated_at。"""
    parts: list[str] = ["updated_at = %s"]
    vals: list[Any] = [time.time()]
    if title is not None:
        parts.append("title = %s")
        vals.append(title)
    vals.append(session_id)
    async with _get_pool().connection() as conn:
        cur = await execute(
            conn,
            f"UPDATE sessions SET {', '.join(parts)} WHERE session_id = %s",
            tuple(vals),
        )
        await conn.commit()
        return cur.rowcount > 0


async def list_sessions() -> list[dict]:
    """按最近更新时间倒序列出所有会话。"""
    async with _get_pool().connection() as conn:
        cur = await execute(conn, "SELECT * FROM sessions ORDER BY updated_at DESC")
        cols = cursor_columns(cur)
        return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def get_session(session_id: str) -> Optional[dict]:
    """按 ID 查询单条会话元数据。"""
    async with _get_pool().connection() as conn:
        cur = await execute(conn, "SELECT * FROM sessions WHERE session_id = %s", (session_id,))
        row = await cur.fetchone()
        if not row:
            return None
        cols = cursor_columns(cur)
        return dict(zip(cols, row))


async def delete_session(session_id: str) -> bool:
    """删除会话记录（checkpoint 需由 sessions 路由另行清理）。"""
    async with _get_pool().connection() as conn:
        await execute(conn, "DELETE FROM session_papers WHERE session_id = %s", (session_id,))
        cur = await execute(conn, "DELETE FROM sessions WHERE session_id = %s", (session_id,))
        await conn.commit()
        return cur.rowcount > 0


async def link_session_paper(session_id: str, paper_id: str) -> None:
    """将会话与已入库 paper_id 关联（上传成功或重复上传时调用）。"""
    now = time.time()
    async with _get_pool().connection() as conn:
        await execute(
            conn,
            "INSERT INTO session_papers (session_id, paper_id, linked_at) "
            "VALUES (%s, %s, %s) ON CONFLICT (session_id, paper_id) DO NOTHING",
            (session_id, paper_id, now),
        )
        await execute(conn, "UPDATE sessions SET updated_at = %s WHERE session_id = %s", (now, session_id))
        await conn.commit()


async def get_session_paper_ids(session_id: str) -> list[str]:
    """返回本会话绑定的 paper_id 列表（检索 scope）。"""
    async with _get_pool().connection() as conn:
        cur = await execute(
            conn,
            "SELECT paper_id FROM session_papers WHERE session_id = %s ORDER BY linked_at",
            (session_id,),
        )
        return [row[0] for row in await cur.fetchall()]


# ── 文件元数据 CRUD ────────────────────────────────────────────────

async def add_file(
    file_id: str, filename: str, paper_id: str,
    size_bytes: int = 0, page_count: int = 0, chunk_count: int = 0,
) -> dict:
    """插入或更新文件记录（同一 file_id 重复上传时 UPSERT）。"""
    now = time.time()
    async with _get_pool().connection() as conn:
        await execute(
            conn,
            "INSERT INTO files (file_id, filename, paper_id, size_bytes, page_count, chunk_count, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (file_id) DO UPDATE SET "
            "filename=EXCLUDED.filename, paper_id=EXCLUDED.paper_id, "
            "size_bytes=EXCLUDED.size_bytes, page_count=EXCLUDED.page_count, "
            "chunk_count=EXCLUDED.chunk_count",
            (file_id, filename, paper_id, size_bytes, page_count, chunk_count, now),
        )
        await conn.commit()
    return {
        "file_id": file_id, "filename": filename, "paper_id": paper_id,
        "size_bytes": size_bytes, "page_count": page_count,
        "chunk_count": chunk_count, "created_at": now,
    }


async def list_files() -> list[dict]:
    """列出已入库 PDF 元数据。"""
    async with _get_pool().connection() as conn:
        cur = await execute(conn, "SELECT * FROM files ORDER BY created_at DESC")
        cols = cursor_columns(cur)
        return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def get_file(file_id: str) -> Optional[dict]:
    """按 file_id 查询文件记录。"""
    async with _get_pool().connection() as conn:
        cur = await execute(conn, "SELECT * FROM files WHERE file_id = %s", (file_id,))
        row = await cur.fetchone()
        if not row:
            return None
        cols = cursor_columns(cur)
        return dict(zip(cols, row))


async def delete_file_record(file_id: str) -> Optional[dict]:
    """删除文件记录并返回被删行（供路由层清理 Milvus / 磁盘）。"""
    async with _get_pool().connection() as conn:
        cur = await execute(conn, "SELECT * FROM files WHERE file_id = %s", (file_id,))
        row = await cur.fetchone()
        if not row:
            return None
        cols = cursor_columns(cur)
        record = dict(zip(cols, row))
        await execute(conn, "DELETE FROM files WHERE file_id = %s", (file_id,))
        await conn.commit()
        return record


async def clear_all_files() -> int:
    """清空 files 表（配合 manage 路由清空向量库时使用）。"""
    async with _get_pool().connection() as conn:
        cur = await execute(conn, "DELETE FROM files")
        await conn.commit()
        return cur.rowcount
