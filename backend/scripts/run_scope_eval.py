"""
读取 scope_eval_dataset.json，直接跑 Agent 图（与线上一致的 LLM + Retriever 配置）。

前提：Milvus 中已有 eval_* 文档；Ollama 已启动。无需启动 run.py。

用法（backend 目录）:
  .\\.venv\\Scripts\\python.exe scripts\\run_scope_eval.py --limit 3
  .\\.venv\\Scripts\\python.exe scripts\\run_scope_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

DATASET = _BACKEND / "eval" / "fixtures" / "scope_eval_dataset.json"
DEFAULT_OUT = _BACKEND / "eval" / "results" / "scope_eval_latest.json"


def _check_must_contain(answer: str, tokens: list[str]) -> tuple[bool, list[str]]:
    lower = (answer or "").lower()
    missing = [t for t in tokens if t.lower() not in lower and t not in (answer or "")]
    return len(missing) == 0, missing


def _scope_ok(paper_ids_hit: list[str], expected: str) -> bool:
    return paper_ids_hit == [expected] or (
        len(paper_ids_hit) == 1 and paper_ids_hit[0] == expected
    )


def _check_ollama() -> None:
    import httpx
    from config import Config

    root = Config.LLM_BASE_URL.replace("/v1", "").rstrip("/")
    try:
        httpx.get(f"{root}/api/tags", timeout=5.0).raise_for_status()
    except Exception as e:
        raise SystemExit(f"Ollama not reachable at {root}: {e}") from e


async def _run_eval(cases: list[dict]) -> dict:
    from agent.graph import build_graph
    from agent.states import fresh_turn_state
    from app.dependencies import _build_llm, ensure_retriever_tool
    from rag.citation import CitationExtractor
    from rag.chat_trace import start_trace, finish_trace
    from config import Config

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )
    logging.getLogger("langchain_milvus").setLevel(logging.ERROR)
    if Config.CHAT_TRACE:
        print(
            f"CHAT_TRACE on — JSON under {Config.CHAT_TRACE_DIR}/eval_<case_id>/",
            flush=True,
        )

    _check_ollama()

    print("Loading LLM + Retriever (same config as run.py)…", flush=True)
    llm = _build_llm()
    retriever = ensure_retriever_tool()
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        citation_extractor=CitationExtractor,
        max_retries=Config.MAX_RETRIES,
        checkpointer=None,
    )
    print("Graph ready.\n", flush=True)

    results = []
    passed = 0
    total = len(cases)

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        query = case["query"]
        paper_id = case["paper_id"]
        must = case.get("must_contain", [])

        print(f"[{i}/{total}] {cid}", flush=True)
        print(f"  paper: {paper_id}", flush=True)
        print(f"  Q: {query}", flush=True)

        trace_id = f"eval_{cid}"
        session_id = f"eval_{cid}"
        t0 = time.time()
        try:
            start_trace(session_id, query, trace_id=trace_id)
            out = await graph.ainvoke(
                fresh_turn_state(query, trace_id, paper_ids=[paper_id])
            )
            finish_trace(trace_id)
            answer = str(out.get("answer", ""))
            citations = out.get("citations") or []
            paper_ids_hit = sorted(
                {c.get("paper_id") for c in citations if c.get("paper_id")}
            )
            ok, missing = _check_must_contain(answer, must)
            scope_ok = _scope_ok(paper_ids_hit, paper_id)
            elapsed = time.time() - t0
            if ok:
                passed += 1
            row = {
                "id": cid,
                "paper_id": paper_id,
                "query": query,
                "passed": ok,
                "scope_ok": scope_ok,
                "missing_tokens": missing,
                "paper_ids_in_citations": paper_ids_hit,
                "elapsed_s": round(elapsed, 1),
                "answer_preview": answer[:400],
                "reference_answer": case.get("reference_answer", ""),
            }
            print(
                f"  {'PASS' if ok else 'FAIL'} scope={'OK' if scope_ok else 'BAD'} {elapsed:.1f}s",
                flush=True,
            )
            if missing:
                print(f"  missing: {missing}", flush=True)
            if not scope_ok:
                print(f"  citation papers: {paper_ids_hit}", flush=True)
        except Exception as e:
            finish_trace(trace_id)
            row = {
                "id": cid,
                "paper_id": paper_id,
                "query": query,
                "passed": False,
                "error": str(e),
            }
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
        results.append(row)

    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scope eval dataset through the agent graph.")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N cases")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    print("ScholarRAG scope eval", flush=True)
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.limit:
        cases = cases[: args.limit]

    try:
        summary = asyncio.run(_run_eval(cases))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\n=== {summary['passed']}/{summary['total']} passed "
        f"({summary['pass_rate']*100:.1f}%) ===",
        flush=True,
    )
    print(f"Report: {out_path}", flush=True)


if __name__ == "__main__":
    main()
