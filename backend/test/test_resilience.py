"""
resilience 模块单元测试（不依赖 LLM / Milvus）。

运行：cd backend && python -m pytest test/test_resilience.py -v
或：python test/test_resilience.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.resilience import (
    invoke_with_timeout,
    is_failed_sub_answer,
    make_failed_sub_answer,
    make_sub_answer,
)
from agent.nodes import _collect_valid_evidence, _format_failed_sub_queries


def test_make_failed_sub_answer():
    sa = make_failed_sub_answer("子问题 A", "TimeoutError: timed out")
    assert sa["query"] == "子问题 A"
    assert sa["answer"] == ""
    assert sa["citations"] == []
    assert sa["status"] == "failed"
    assert "Timeout" in sa["error"]
    assert is_failed_sub_answer(sa)


def test_collect_valid_evidence_skips_failed():
    ok = make_sub_answer("q1", "A" * 40 + " [1]", [{"chunk_id": "c1"}])
    bad = make_failed_sub_answer("q2", "network error")
    valid = _collect_valid_evidence([ok, bad])
    assert len(valid) == 1
    assert valid[0]["query"] == "q1"


def test_format_failed_sub_queries():
    note, failed = _format_failed_sub_queries([
        make_sub_answer("ok", "answer " * 10, []),
        make_failed_sub_answer("failed q", "Milvus down"),
    ])
    assert len(failed) == 1
    assert failed[0]["query"] == "failed q"
    assert "Unavailable Evidence" in note
    assert "Milvus down" in note

    empty_note, empty_list = _format_failed_sub_queries([make_sub_answer("ok", "x", [])])
    assert empty_note == ""
    assert empty_list == []


async def _slow_coro():
    await asyncio.sleep(0.2)
    return {"answer": "done"}


async def _fast_coro():
    return {"answer": "ok"}


async def _boom_coro():
    raise RuntimeError("boom")


def test_invoke_with_timeout_success():
    result, err = asyncio.run(invoke_with_timeout(_fast_coro(), timeout_sec=1.0, label="fast"))
    assert err is None
    assert result is not None
    assert result["answer"] == "ok"


def test_invoke_with_timeout_times_out():
    result, err = asyncio.run(invoke_with_timeout(_slow_coro(), timeout_sec=0.05, label="slow"))
    assert result is None
    assert err is not None
    assert "timed out" in err


def test_invoke_with_timeout_exception():
    result, err = asyncio.run(invoke_with_timeout(_boom_coro(), timeout_sec=1.0, label="boom"))
    assert result is None
    assert "RuntimeError" in (err or "")


if __name__ == "__main__":
    test_make_failed_sub_answer()
    test_collect_valid_evidence_skips_failed()
    test_format_failed_sub_queries()
    test_invoke_with_timeout_success()
    test_invoke_with_timeout_times_out()
    test_invoke_with_timeout_exception()
    print("All resilience tests passed.")
