"""
打印单题检索 Top-K，便于在 scope_eval_dataset.json 中填写 relevant_ids。

用法（backend 目录）:
  .\\.venv\\Scripts\\python.exe scripts\\retrieval_eval_preview.py --id eval_01_abstract_efficiency
  .\\.venv\\Scripts\\python.exe scripts\\retrieval_eval_preview.py --limit 5
  .\\.venv\\Scripts\\python.exe scripts\\retrieval_eval_preview.py --id eval_01_abstract_efficiency
  .\\.venv\\Scripts\\python.exe scripts\\retrieval_eval_preview.py --single-paper --id eval_01_abstract_efficiency
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

DATASET = _BACKEND / "eval" / "fixtures" / "scope_eval_dataset.json"

from config import Config  # noqa: E402
from eval.retrieval_eval_runner import all_eval_paper_ids  # noqa: E402


def _preview_case(case: dict, k: int, fetch_k: int, multi_paper: bool, all_papers: list[str]) -> None:
    from app.dependencies import ensure_retriever_tool

    retriever = ensure_retriever_tool()._retriever
    paper_filter = all_papers if multi_paper else [case["paper_id"]]

    print("=" * 88)
    print(f"id:       {case['id']}")
    print(f"paper_id: {case['paper_id']}")
    print(f"query:    {case['query']}")
    mode = f"session ({len(paper_filter)} papers)" if multi_paper else "single-paper"
    print(f"filter:   {mode} → {paper_filter}")
    print(f"must:     {case.get('must_contain', [])}")
    print("-" * 88)

    docs = retriever.retrieve(
        case["query"],
        k=k,
        fetch_k=fetch_k,
        rerank=True,
        expand_parent=True,
        rrf_k=Config.RRF_K,
        paper_id_filter=paper_filter,
    )

    for i, doc in enumerate(docs, 1):
        cid = doc.metadata.get("chunk_id", "N/A")
        pid = doc.metadata.get("paper_id", "N/A")
        page = doc.metadata.get("page_num", "?")
        section = doc.metadata.get("section_path", "")
        text = doc.page_content[:280].replace("\n", " ")
        if pid == case["paper_id"]:
            mark = ""
        elif multi_paper:
            mark = "  *** OTHER PAPER (noise if ranked high) ***"
        else:
            mark = ""
        print(f"\n[{i}] chunk_id: {cid}")
        print(f"    paper_id: {pid}{mark}")
        print(f"    page: {page}  section: {section[:60]}")
        print(f"    text: {text}...")

    print(
        "\nLabel scope_eval_dataset.json → relevant_ids with chunk_id(s) from "
        f"paper_id={case['paper_id']} only (ignore *** OTHER PAPER *** rows).\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview retrieval hits for labeling")
    parser.add_argument("--id", type=str, default=None, help="Single case id")
    parser.add_argument("--limit", type=int, default=None, help="First N cases")
    parser.add_argument(
        "--single-paper",
        action="store_true",
        help="Only search case paper_id (default: all eval PDFs in one session, like UI)",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of results to show")
    parser.add_argument("--fetch-k", type=int, default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("langchain_milvus").setLevel(logging.ERROR)

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
        if not cases:
            raise SystemExit(f"Unknown id: {args.id}")
    elif args.limit:
        cases = cases[: args.limit]

    fetch_k = args.fetch_k or Config.FETCH_K
    all_papers = all_eval_paper_ids(data["cases"])

    multi_paper = not args.single_paper
    for case in cases:
        _preview_case(case, args.k, fetch_k, multi_paper, all_papers)


if __name__ == "__main__":
    main()
