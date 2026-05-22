"""psycopg 异步执行辅助（集中处理 Pyright 与动态 SQL 的类型检查）。"""

from typing import Any, Sequence

from psycopg import AsyncConnection
from psycopg import sql


async def execute(
    conn: AsyncConnection,
    query: str,
    params: Sequence[Any] | None = None,
):
    """执行参数化 SQL；psycopg stubs 对 str query 过严，运行时完全合法。"""
    if params is None:
        return await conn.execute(query)  # type: ignore[arg-type]
    return await conn.execute(query, params)  # type: ignore[arg-type]


async def create_database(conn: AsyncConnection, db_name: str) -> None:
    """安全创建数据库（标识符用 sql.Identifier，避免 f-string SQL）。"""
    await conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))


def cursor_columns(cur) -> list[str]:
    """从 cursor.description 取列名；无结果集时返回空列表。"""
    if cur.description is None:
        return []
    return [d.name for d in cur.description]
