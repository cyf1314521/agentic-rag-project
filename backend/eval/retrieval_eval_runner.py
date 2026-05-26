"""
检索评测跑批逻辑（供 scripts/run_retrieval_eval.py 调用）。

基于 scope_eval_dataset.json：chunk_id 金标准 + 可选多 paper 干扰模式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from langchain_core.documents import Document

from eval.eval_retrieval import calculate_metrics_from_docs, is_hit_answer
from rag.retrieval import Retriever

RETRIEVAL_CONFIGS: dict[str, dict[str, bool | str]] = {
    "full": {"rerank": True, "expand_parent": True, "search_mode": "hybrid"},
    "no_rerank": {"rerank": False, "expand_parent": True, "search_mode": "hybrid"},
    "no_parent": {"rerank": True, "expand_parent": False, "search_mode": "hybrid"},
    "minimal": {"rerank": False, "expand_parent": False, "search_mode": "hybrid"},
    # 混合 vs 单路检索（其余与 full 相同：rerank + parent）
    "channel_hybrid": {"rerank": True, "expand_parent": True, "search_mode": "hybrid"},
    "channel_dense": {"rerank": True, "expand_parent": True, "search_mode": "dense"},
    "channel_bm25": {"rerank": True, "expand_parent": True, "search_mode": "sparse"},
    # Milvus 单路/融合阶段（无 CrossEncoder，便于看通道本身差异）
    "channel_hybrid_raw": {"rerank": False, "expand_parent": True, "search_mode": "hybrid"},
    "channel_dense_raw": {"rerank": False, "expand_parent": True, "search_mode": "dense"},
    "channel_bm25_raw": {"rerank": False, "expand_parent": True, "search_mode": "sparse"},
}

CHANNEL_COMPARISON_CONFIGS = (
    "channel_hybrid",
    "channel_dense",
    "channel_bm25",
    "channel_hybrid_raw",
    "channel_dense_raw",
    "channel_bm25_raw",
)


@dataclass(frozen=True)
class EvalSettings:
    k: int = 5
    fetch_k: int = 20
    rrf_k: int = 60


def _contamination_rate(docs: list[Document], expected_paper_id: str, k: int) -> float:
    """Top-k 中 paper_id 不等于本题的比例。"""
    top = docs[:k]
    if not top:
        return 0.0
    wrong = sum(1 for d in top if d.metadata.get("paper_id") != expected_paper_id)
    return wrong / len(top)


def run_case_retrieval(
    retriever: Retriever,
    case: dict,
    config: Mapping[str, bool | str],
    settings: EvalSettings,
    paper_id_filter: list[str] | None,
    use_llm_hit: bool = False,
    llm=None,
) -> dict[str, Any]:
    """对单题执行一次 retrieve 并返回指标与明细。"""
    query = case["query"]
    expected_paper = case["paper_id"]
    paper_id_filter = paper_id_filter or [expected_paper]

    docs = retriever.retrieve(
        query,
        k=settings.k,
        fetch_k=settings.fetch_k,
        rerank=bool(config["rerank"]),
        expand_parent=bool(config["expand_parent"]),
        rrf_k=settings.rrf_k,
        paper_id_filter=paper_id_filter,
        search_mode=str(config.get("search_mode", "hybrid")),
    )
    retrieved_ids = [str(d.metadata.get("chunk_id", "")) for d in docs]
    row: dict[str, Any] = {
        "retrieved_ids": retrieved_ids,
        "retrieved_paper_ids": [d.metadata.get("paper_id") for d in docs[: settings.k]],
        "contamination@k": round(_contamination_rate(docs, expected_paper, settings.k), 4),
        "num_retrieved": len(docs),
    }

    if use_llm_hit:
        hits = [is_hit_answer(doc, case, llm) for doc in docs]
        num_relevant = max(len(case.get("relevant_ids") or []), max(settings.k // 2, 1))
        from eval.eval_retrieval import calculate_metrics_from_hits

        metrics = calculate_metrics_from_hits(hits, num_relevant, settings.k)
        row["hit_flags"] = hits[: settings.k]
        row["mode"] = "llm_hit"
    else:
        relevant_ids = case.get("relevant_ids") or []
        if not relevant_ids:
            row["skipped"] = True
            row["skip_reason"] = "empty relevant_ids"
            return row
        metrics = calculate_metrics_from_docs(docs, relevant_ids, settings.k)
        row["mode"] = "chunk_id"
        row["relevant_ids"] = relevant_ids

    row["recall"] = round(metrics["recall"], 4)
    row["precision"] = round(metrics["precision"], 4)
    row["mrr"] = round(metrics["mrr"], 4)
    row["map"] = round(metrics["ap"], 4)
    return row


def aggregate_metrics(per_case_rows: list[dict], k: int) -> dict[str, Any]:
    """对未 skipped 的 case 行求平均。"""
    usable = [r for r in per_case_rows if not r.get("skipped")]
    if not usable:
        return {
            f"recall@{k}": 0.0,
            f"precision@{k}": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "contamination@k": 0.0,
            "num_queries": 0,
            "num_skipped": len(per_case_rows),
        }

    def _avg(key: str) -> float:
        return sum(r[key] for r in usable) / len(usable)

    return {
        f"recall@{k}": round(_avg("recall"), 4),
        f"precision@{k}": round(_avg("precision"), 4),
        "mrr": round(_avg("mrr"), 4),
        "map": round(_avg("map"), 4),
        "contamination@k": round(_avg("contamination@k"), 4),
        "num_queries": len(usable),
        "num_skipped": len(per_case_rows) - len(usable),
    }


def all_eval_paper_ids(cases: list[dict]) -> list[str]:
    """数据集中全部 paper_id（去重保序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for c in cases:
        pid = c.get("paper_id")
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out
