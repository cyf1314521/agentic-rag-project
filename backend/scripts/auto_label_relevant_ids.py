"""
在 10 篇 eval PDF 同会话范围内检索，自动填写 scope_eval_dataset.json 的 relevant_ids。

规则（仅采纳本题 paper_id 的 chunk）：
1. 优先：正文包含 must_contain 全部关键词
2. 其次：包含至少一半 must_contain
3. 兜底：该题在全局排序中名次最高的本题 chunk（最多 3 个）

用法（backend 目录）:
  .\\.venv\\Scripts\\python.exe scripts\\auto_label_relevant_ids.py
  .\\.venv\\Scripts\\python.exe scripts\\auto_label_relevant_ids.py --dry-run
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


def _token_in_text(token: str, text: str) -> bool:
    if not token or not text:
        return False
    if token in text:
        return True
    return token.lower() in text.lower()


def _must_contain_score(tokens: list[str], text: str) -> tuple[int, int]:
    if not tokens:
        return 0, 0
    hits = sum(1 for t in tokens if _token_in_text(t, text))
    return hits, len(tokens)


def _pick_relevant_ids(docs: list, paper_id: str, must: list[str], max_ids: int = 3) -> list[str]:
    """从检索结果中选出 relevant chunk_id 列表。"""
    target = [d for d in docs if d.metadata.get("paper_id") == paper_id]
    if not target:
        return []

    scored: list[tuple[int, int, int, str]] = []
    for rank, doc in enumerate(target, 1):
        cid = str(doc.metadata.get("chunk_id", ""))
        if not cid:
            continue
        hits, total = _must_contain_score(must, doc.page_content)
        scored.append((hits, total, -rank, cid))

    scored.sort(key=lambda x: (x[0] / x[1] if x[1] else 0, x[0], x[2]), reverse=True)

    chosen: list[str] = []
    seen: set[str] = set()

    def _add(cid: str) -> None:
        if cid and cid not in seen:
            seen.add(cid)
            chosen.append(cid)

    for hits, total, _, cid in scored:
        if total and hits == total:
            _add(cid)
    if len(chosen) < max_ids:
        for hits, total, _, cid in scored:
            if total and hits >= max(1, (total + 1) // 2) and hits < total:
                _add(cid)
            if len(chosen) >= max_ids:
                break
    if not chosen and scored:
        _add(scored[0][3])

    return chosen[:max_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-fill relevant_ids in scope_eval_dataset.json")
    parser.add_argument("--dry-run", action="store_true", help="Print only, do not write JSON")
    parser.add_argument("--preview-k", type=int, default=20, help="Retrieve this many docs per query")
    parser.add_argument("--single-paper", action="store_true", help="Search only case paper_id")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("langchain_milvus").setLevel(logging.ERROR)

    from app.dependencies import ensure_retriever_tool

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    all_papers = all_eval_paper_ids(cases)
    retriever = ensure_retriever_tool()._retriever

    print(f"Auto-label {len(cases)} cases ({'single-paper' if args.single_paper else 'session 10 papers'})…\n")

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        paper_id = case["paper_id"]
        must = case.get("must_contain", [])
        paper_filter = [paper_id] if args.single_paper else all_papers

        docs = retriever.retrieve(
            case["query"],
            k=args.preview_k,
            fetch_k=Config.FETCH_K,
            rerank=True,
            expand_parent=True,
            rrf_k=Config.RRF_K,
            paper_id_filter=paper_filter,
        )

        ids = _pick_relevant_ids(docs, paper_id, must)
        case["relevant_ids"] = ids

        global_top = docs[0].metadata.get("paper_id") if docs else None
        contam = sum(1 for d in docs[: Config.TOP_K] if d.metadata.get("paper_id") != paper_id)
        print(f"[{i:02d}/{len(cases)}] {cid}")
        print(f"  relevant_ids ({len(ids)}): {ids}")
        print(f"  top1 paper: {global_top}  contamination@5≈{contam}/{Config.TOP_K}")

    if args.dry_run:
        print("\n(dry-run, JSON not written)")
        return

    data["description"] = (
        "Session-scope + retrieval eval. relevant_ids auto-labeled via "
        "scripts/auto_label_relevant_ids.py (10-paper session retrieve)."
    )
    DATASET.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nUpdated {DATASET}")


if __name__ == "__main__":
    main()
