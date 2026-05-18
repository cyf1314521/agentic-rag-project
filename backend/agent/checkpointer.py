"""
LangGraph Checkpointer 工厂。

Checkpointer 负责将图状态（含 messages）按 thread_id 持久化，
本项目生产环境使用 PostgreSQL（AsyncPostgresSaver），测试可用内存 MemorySaver。
"""

from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver


def create_memory_checkpointer() -> BaseCheckpointSaver:
    """创建进程内内存检查点（重启丢失，适合单元测试）。"""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


@asynccontextmanager
async def create_postgres_checkpointer(conn_string: str):
    """
    异步上下文管理器：创建 Postgres checkpointer 并执行 setup 建表。

    用法:
        async with create_postgres_checkpointer(uri) as cp:
            graph = build_graph(..., checkpointer=cp)
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
