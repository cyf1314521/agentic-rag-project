"""
读取 scope_eval_dataset.json，直接跑 Agent 图（与线上一致的 LLM + Retriever 配置）。

前提：Milvus 中已有 eval_* 文档；Ollama 已启动。无需启动 run.py。

默认与 UI 一致：一次会话绑定全部 10 篇 eval PDF（paper_id_filter 为全集）。
单篇对照：加 --single-paper。

用法（backend 目录）:
  .\\.venv\\Scripts\\python.exe scripts\\run_scope_eval.py --limit 3
  .\\.venv\\Scripts\\python.exe scripts\\run_scope_eval.py
  .\\.venv\\Scripts\\python.exe scripts\\run_scope_eval.py --single-paper
  .\\.venv\\Scripts\\python.exe scripts\\run_scope_eval.py --compare
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
COMPARE_OUT = _BACKEND / "eval" / "results" / "scope_eval_compare_latest.json"


def _check_must_contain(answer: str, tokens: list[str]) -> tuple[bool, list[str]]:
    lower = (answer or "").lower()
    missing = [t for t in tokens if t.lower() not in lower and t not in (answer or "")]
    return len(missing) == 0, missing


def _check_ollama() -> None:
    import httpx
    from config import Config

    root = Config.LLM_BASE_URL.replace("/v1", "").rstrip("/")
    try:
        httpx.get(f"{root}/api/tags", timeout=5.0).raise_for_status()
    except Exception as e:
        raise SystemExit(f"Ollama not reachable at {root}: {e}") from e


async def _run_eval(cases: list[dict], multi_paper: bool) -> dict:
    from agent.graph import build_graph
    from agent.states import fresh_turn_state
    from app.dependencies import _build_llm, ensure_retriever_tool
    from eval.retrieval_eval_runner import all_eval_paper_ids
    from rag.citation import (
        CitationExtractor,
        paper_ids_from_citations,
        parse_citation_indices,
        scope_ok_for_paper,
    )
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

    session_paper_ids = all_eval_paper_ids(cases) if multi_paper else None
    if multi_paper and session_paper_ids:
        print(f"Session scope: {len(session_paper_ids)} eval papers (same as UI multi-PDF chat)\n", flush=True)

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
        scope_paper_ids = session_paper_ids if multi_paper else [paper_id]
        t0 = time.time()
        try:
            start_trace(session_id, query, trace_id=trace_id)
            out = await graph.ainvoke(
                fresh_turn_state(query, trace_id, paper_ids=scope_paper_ids)
            )
            finish_trace(trace_id)
            answer = str(out.get("answer", ""))
            citations = out.get("citations") or []
            citation_indices = parse_citation_indices(answer)
            paper_ids_cited = paper_ids_from_citations(answer, citations)
            paper_ids_pool = sorted(
                {str(c.get("paper_id")) for c in citations if c.get("paper_id")}
            )
            ok, missing = _check_must_contain(answer, must)
            scope_ok = scope_ok_for_paper(paper_ids_cited, paper_id)
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
                "citation_indices_in_answer": citation_indices,
                "paper_ids_cited": paper_ids_cited,
                "paper_ids_in_retrieval_pool": paper_ids_pool,
                "session_paper_ids": scope_paper_ids,
                "paper_filter_mode": "multi_paper" if multi_paper else "single_paper",
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
                print(f"  cited papers: {paper_ids_cited} (indices {citation_indices})", flush=True)
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

    scope_ok = sum(1 for r in results if r.get("scope_ok"))
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0,
        "scope_ok": scope_ok,
        "scope_ok_rate": round(scope_ok / total, 3) if total else 0,
        "paper_filter_mode": "multi_paper" if multi_paper else "single_paper",
        "session_paper_ids": session_paper_ids or [],
        "results": results,
    }


def _print_summary(label: str, summary: dict) -> None:
    print(
        f"\n=== {label}: {summary['passed']}/{summary['total']} passed "
        f"({summary['pass_rate']*100:.1f}%), "
        f"scope {summary.get('scope_ok', 0)}/{summary['total']} "
        f"({summary.get('scope_ok_rate', 0)*100:.1f}%) ===",
        flush=True,
    )


async def _run_compare(cases: list[dict]) -> dict:
    from datetime import datetime, timezone

    print("Running single-paper (per-case one PDF) eval …", flush=True)
    single = await _run_eval(cases, multi_paper=False)
    print("Running multi-paper (10 PDF session) eval …", flush=True)
    multi = await _run_eval(cases, multi_paper=True)

    multi_by_id = {r["id"]: r for r in multi["results"]}
    per_case_delta = []
    for row in single["results"]:
        cid = row["id"]
        m = multi_by_id.get(cid, {})
        per_case_delta.append(
            {
                "id": cid,
                "paper_id": row.get("paper_id"),
                "multi_passed": m.get("passed"),
                "single_passed": row.get("passed"),
                "multi_scope_ok": m.get("scope_ok"),
                "single_scope_ok": row.get("scope_ok"),
            }
        )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(_BACKEND)).replace("\\", "/"),
        "multi_paper": multi,
        "single_paper": single,
        "delta": {
            "passed": multi["passed"] - single["passed"],
            "pass_rate": round(multi["pass_rate"] - single["pass_rate"], 3),
            "scope_ok": multi.get("scope_ok", 0) - single.get("scope_ok", 0),
            "scope_ok_rate": round(
                multi.get("scope_ok_rate", 0) - single.get("scope_ok_rate", 0),
                3,
            ),
            "note": "negative passed = single better; negative scope_ok = single better",
        },
        "per_case": per_case_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scope eval dataset through the agent graph.")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N cases")
    parser.add_argument(
        "--single-paper",
        action="store_true",
        help="Only scope to case paper_id (default: all eval PDFs, same as one UI session)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run multi-paper and single-paper back-to-back; write scope_eval_compare_latest.json",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    print("ScholarRAG scope eval", flush=True)
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.limit:
        cases = cases[: args.limit]

    try:
        if args.compare:
            summary = asyncio.run(_run_compare(cases))
            out_path = COMPARE_OUT
        else:
            summary = asyncio.run(_run_eval(cases, multi_paper=not args.single_paper))
            out_path = Path(args.out)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.compare:
        _print_summary("Multi-paper", summary["multi_paper"])
        _print_summary("Single-paper", summary["single_paper"])
        print(f"\nDelta passed (multi - single): {summary['delta']['passed']:+d}", flush=True)
        print(f"Delta scope_ok (multi - single): {summary['delta']['scope_ok']:+d}", flush=True)
    else:
        _print_summary(summary["paper_filter_mode"], summary)
    print(f"Report: {out_path}", flush=True)


if __name__ == "__main__":
    main()
