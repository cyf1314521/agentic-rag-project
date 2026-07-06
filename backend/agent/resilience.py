"""
子 Agent 容错工具：超时、异常捕获、Fallback SubAnswer。

设计原则：
- 单路子图失败不应拖死整次 graph.astream
- 失败时返回 status="failed" 的 SubAnswer，主图仍可 prepare_synthesis → synthesize
"""

import asyncio
import logging
from typing import Any, Optional

from .states import SubAnswer

logger = logging.getLogger(__name__)


def make_sub_answer(
    query: str,
    answer: str,
    citations: list[dict],
    *,
    status: str = "ok",
    error: str = "",
) -> SubAnswer:
    """构造一条子 Agent 结果（成功或失败占位）。"""
    return SubAnswer(
        query=query,
        answer=answer,
        citations=citations,
        status=status,
        error=error,
    )


def make_failed_sub_answer(query: str, error: str) -> SubAnswer:
    """子图硬失败时的 Fallback：空答案 + 空引用，带失败原因。"""
    return make_sub_answer(query, "", [], status="failed", error=error)


def is_failed_sub_answer(sa: SubAnswer) -> bool:
    return sa.get("status") == "failed"


async def invoke_with_timeout(
    coro,
    *,
    timeout_sec: float,
    label: str,
) -> tuple[Optional[Any], Optional[str]]:
    """
    在 asyncio 层为子图 ainvoke 加超时。

    Returns:
        (result, None) 成功
        (None, error_message) 超时或其它异常
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_sec)
        return result, None
    except asyncio.TimeoutError:
        msg = f"{label} timed out after {timeout_sec:.0f}s"
        logger.warning(msg)
        return None, msg
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("%s failed", label)
        return None, msg
