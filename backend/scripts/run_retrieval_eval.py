"""
在 scope_eval_dataset.json 上跑检索层评测（chunk_id 金标准 + 消融）。

前提：Milvus 已入库 eval_* PDF；无需启动 run.py / Ollama。

用法（backend 目录）:
  .\\.venv\\Scripts\\python.exe scripts\\retrieval_eval_preview.py --id eval_01_abstract_efficiency
  .\\.venv\\Scripts\\python.exe scripts\\run_retrieval_eval.py --limit 3
  .\\.venv\\Scripts\\python.exe scripts\\run_retrieval_eval.py
  .\\.venv\\Scripts\\python.exe scripts\\run_retrieval_eval.py
  .\\.venv\\Scripts\\python.exe scripts\\run_retrieval_eval.py --single-paper
  .\\.venv\\Scripts\\python.exe scripts\\run_retrieval_eval.py --mode llm
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import logging
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

DATASET = _BACKEND / "eval" / "fixtures" / "scope_eval_dataset.json"
DEFAULT_OUT = _BACKEND / "eval" / "results" / "retrieval_eval_latest.json"
SCOPE_RESULTS = _BACKEND / "eval" / "results" / "scope_eval_latest.json"

from config import Config  # noqa: E402
from eval.retrieval_eval_runner import (  # noqa: E402
    CHANNEL_COMPARISON_CONFIGS,
    RETRIEVAL_CONFIGS,
    EvalSettings,
    aggregate_metrics,
    all_eval_paper_ids,
    run_case_retrieval,
)


def _load_cases(limit: int | None) -> list[dict]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    if limit:
        cases = cases[:limit]
    return cases


def _merge_scope_alignment(per_case: list[dict]) -> None:
    """若存在 scope_eval_latest.json，按 id 附上端到端 passed。"""
    if not SCOPE_RESULTS.is_file():
        return
    scope = json.loads(SCOPE_RESULTS.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in scope.get("results", [])}
    for row in per_case:
        cid = row.get("id")
        if cid in by_id:
            row["scope_passed"] = by_id[cid].get("passed")
            row["scope_missing_tokens"] = by_id[cid].get("missing_tokens", [])


def run_eval(
    cases: list[dict],
    config_names: Sequence[str],
    settings: EvalSettings,
    multi_paper: bool,
    use_llm_hit: bool,
) -> dict:
    from app.dependencies import ensure_retriever_tool, _build_llm

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )
    logging.getLogger("langchain_milvus").setLevel(logging.ERROR)

    print("Loading Retriever …", flush=True)
    retriever = ensure_retriever_tool()._retriever
    llm = _build_llm() if use_llm_hit else None

    session_paper_ids: list[str] | None = None
    if multi_paper:
        session_paper_ids = all_eval_paper_ids(cases)
        print(f"Multi-paper filter: {len(session_paper_ids)} papers", flush=True)

    summary_configs: dict[str, dict] = {}
    per_case_out: list[dict] = []

    for config_name in config_names:
        if config_name not in RETRIEVAL_CONFIGS:
            raise SystemExit(f"Unknown config: {config_name}. Choose from {list(RETRIEVAL_CONFIGS)}")
        config = RETRIEVAL_CONFIGS[config_name]
        print(f"\n=== Config: {config_name} {config} ===", flush=True)
        case_rows: list[dict] = []

        for i, case in enumerate(cases, 1):
            cid = case["id"]
            paper_filter = session_paper_ids if multi_paper else [case["paper_id"]]
            print(f"[{i}/{len(cases)}] {cid}", flush=True)
            try:
                row = run_case_retrieval(
                    retriever,
                    case,
                    config,
                    settings,
                    paper_id_filter=paper_filter,
                    use_llm_hit=use_llm_hit,
                    llm=llm,
                )
            except Exception as e:
                row = {"skipped": True, "skip_reason": str(e), "error": True}
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()

            if row.get("skipped"):
                print(f"  SKIP: {row.get('skip_reason', 'unknown')}", flush=True)
            else:
                print(
                    f"  recall={row['recall']:.2%} mrr={row['mrr']:.4f} "
                    f"contamination={row.get('contamination@k', 0):.2%}",
                    flush=True,
                )

            case_entry = {"id": cid, "paper_id": case["paper_id"], "configs": {config_name: row}}
            # merge into per_case_out by id
            existing = next((p for p in per_case_out if p["id"] == cid), None)
            if existing:
                existing["configs"][config_name] = row
            else:
                per_case_out.append(case_entry)

            case_rows.append(row)

        summary_configs[config_name] = {
            **aggregate_metrics(case_rows, settings.k),
            "retrieve_options": config,
            "paper_filter_mode": "multi_paper" if multi_paper else "single",
        }

    _merge_scope_alignment(per_case_out)

    labeled = sum(1 for c in cases if c.get("relevant_ids"))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(_BACKEND)).replace("\\", "/"),
        "mode": "llm_hit" if use_llm_hit else "chunk_id",
        "settings": {
            "k": settings.k,
            "fetch_k": settings.fetch_k,
            "rrf_k": settings.rrf_k,
        },
        "cases_total": len(cases),
        "cases_with_relevant_ids": labeled,
        "configs": summary_configs,
        "per_case": per_case_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval eval on scope_eval_dataset.json")
    parser.add_argument("--limit", type=int, default=None, help="Only first N cases")
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated config names (see --preset)",
    )
    parser.add_argument(
        "--preset",
        choices=("pipeline", "channel", "all"),
        default="pipeline",
        help="pipeline=重排/父块消融; channel=hybrid vs dense vs BM25; all=全部",
    )
    parser.add_argument(
        "--single-paper",
        action="store_true",
        help="Only search case paper_id (default: all eval PDFs, same as one UI session)",
    )
    parser.add_argument(
        "--mode",
        choices=("chunk_id", "llm"),
        default="chunk_id",
        help="chunk_id needs relevant_ids; llm uses reference_answer",
    )
    parser.add_argument("--k", type=int, default=None, help=f"Top-k (default {Config.TOP_K})")
    parser.add_argument("--fetch-k", type=int, default=None, help=f"Fetch-k (default {Config.FETCH_K})")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if args.mode == "chunk_id":
        cases_preview = _load_cases(args.limit)
        unlabeled = [c["id"] for c in cases_preview if not c.get("relevant_ids")]
        if unlabeled:
            print(
                f"Warning: {len(unlabeled)} case(s) have empty relevant_ids — "
                "they will be skipped in chunk_id mode. Label via:\n"
                "  scripts/retrieval_eval_preview.py --id <case_id>\n",
                flush=True,
            )

    settings = EvalSettings(
        k=args.k or Config.TOP_K,
        fetch_k=args.fetch_k or Config.FETCH_K,
        rrf_k=Config.RRF_K,
    )
    if args.configs:
        config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    elif args.preset == "channel":
        config_names = list(CHANNEL_COMPARISON_CONFIGS)
    elif args.preset == "all":
        config_names = list(RETRIEVAL_CONFIGS.keys())
    else:
        config_names = ["full", "no_rerank", "no_parent", "minimal"]
    cases = _load_cases(args.limit)

    print("ScholarRAG retrieval eval", flush=True)
    print(f"Dataset: {DATASET.name} ({len(cases)} cases)", flush=True)

    try:
        summary = run_eval(
            cases,
            config_names,
            settings,
            multi_paper=not args.single_paper,
            use_llm_hit=args.mode == "llm",
        )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nReport: {out_path}", flush=True)
    for name, agg in summary["configs"].items():
        print(
            f"  {name}: recall@{settings.k}={agg.get(f'recall@{settings.k}', 0):.2%} "
            f"mrr={agg.get('mrr', 0):.4f} "
            f"queries={agg.get('num_queries', 0)} skipped={agg.get('num_skipped', 0)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
