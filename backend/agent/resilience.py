"""
子 Agent 容错工具：超时、重试、请求级 deadline、Fallback SubAnswer。

设计原则：
- 单路子图失败不应拖死整次 graph.astream
- 节点内对瞬时错误（5xx/超时）做有界重试
- 整请求 deadline 约束子图与各步超时
"""

import asyncio
import logging
import random
import time
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Optional, Sequence, TypeVar

from .states import SubAnswer

logger = logging.getLogger(__name__)

T = TypeVar("T")

_request_deadline: ContextVar[Optional[float]] = ContextVar("request_deadline", default=None)

# LangChain / OpenAI / httpx 常见可重试异常名
_RETRYABLE_TYPE_NAMES = frozenset({
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "ConnectError",
    "ReadTimeout",
    "ConnectTimeout",
    "RemoteProtocolError",
})


def parse_backoff_ms(value: str, default: Sequence[int]) -> tuple[int, ...]:
    """解析 '200,500,1000' 形式的重试退避配置。"""
    try:
        parts = [int(x.strip()) for x in value.split(",") if x.strip()]
        return tuple(parts) if parts else tuple(default)
    except ValueError:
        return tuple(default)


def set_request_deadline(seconds: float) -> None:
    """在 chat 入口设置整请求 monotonic deadline。"""
    _request_deadline.set(time.monotonic() + max(0.1, seconds))


def clear_request_deadline() -> None:
    _request_deadline.set(None)


def remaining_seconds() -> Optional[float]:
    """距离整请求 deadline 的剩余秒数；未设置则 None。"""
    deadline = _request_deadline.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def effective_timeout(configured_sec: float) -> float:
    """取配置超时与剩余 deadline 的较小值。"""
    rem = remaining_seconds()
    if rem is None:
        return configured_sec
    return max(0.1, min(configured_sec, rem))


def is_retryable_error(exc: BaseException) -> bool:
    """瞬时网络/限流/5xx 可重试；业务错误（4xx 除 429）不重试。"""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    if type(exc).__name__ in _RETRYABLE_TYPE_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status in (429, 502, 503, 504):
        return True
    return False


def _sleep_seconds(backoff_ms: Sequence[int], attempt: int) -> float:
    idx = min(attempt - 1, len(backoff_ms) - 1)
    base = backoff_ms[idx] / 1000.0
    return base + random.uniform(0, base * 0.2)


def retry_sync(
    fn: Callable[[], T],
    *,
    label: str,
    max_attempts: int,
    backoff_ms: Sequence[int],
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """同步调用重试（用于 run_in_executor 内的 Milvus 检索）。"""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt >= max_attempts or not is_retryable_error(e):
                raise
            if on_retry:
                on_retry(attempt, e)
            delay = _sleep_seconds(backoff_ms, attempt)
            logger.warning("%s retry %d/%d after %s (sleep %.2fs)", label, attempt, max_attempts, e, delay)
            time.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{label}: retry_sync failed without exception")


async def retry_async(
    factory: Callable[[], Awaitable[T]],
    *,
    label: str,
    max_attempts: int,
    backoff_ms: Sequence[int],
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """异步调用重试（用于 LLM ainvoke）。"""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            return await factory()
        except Exception as e:
            last_exc = e
            if attempt >= max_attempts or not is_retryable_error(e):
                raise
            if on_retry:
                on_retry(attempt, e)
            delay = _sleep_seconds(backoff_ms, attempt)
            logger.warning("%s retry %d/%d after %s (sleep %.2fs)", label, attempt, max_attempts, e, delay)
            await asyncio.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{label}: retry_async failed without exception")


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

    使用 create_task + wait，避免子图内部的 TimeoutError 被误判为「外层超时」。

    Returns:
        (result, None) 成功
        (None, error_message) 超时或其它异常
    """
    budget = effective_timeout(timeout_sec)
    task = asyncio.create_task(coro)
    done, _pending = await asyncio.wait({task}, timeout=budget)
    if task not in done:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        msg = f"{label} timed out after {budget:.0f}s"
        logger.warning(msg)
        return None, msg
    try:
        return task.result(), None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("%s failed", label)
        return None, msg
