"""
检索评测模块（与具体数据集解耦）。

指标：Recall@k、Precision@k、MRR、MAP。
支持三种命中判定：chunk_id 精确匹配、LLM 判断 chunk 是否含答案、MMDocIR 页码匹配。

Test case formats:
    1. Chunk ID matching (legacy, not recommended):
       {"query": str, "relevant_ids": list[str]}

    2. LLM-based answer matching (recommended for custom datasets):
       {"query": str, "reference_answer": str}
       Use with hit_fn=is_hit_answer for chunk-agnostic evaluation.

    3. Page-based matching (MMDocIR):
       {"query": str, "relevant_pages": list[int], "paper_id": str, ...}
       Use with hit_fn=is_hit_page from mmdocir_adapter.

For custom hit functions, signature: hit_fn(doc: Document, case: dict) -> bool
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from rag.retrieval import Retriever

HitFn = Callable[[Document, dict], bool]


def _message_text(content: str | list[str | dict[str, Any]]) -> str:
    """Normalize AIMessage.content (str or multimodal blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def is_hit_answer(doc: Document, case: dict, llm: ChatOpenAI | None = None) -> bool:
    """LLM-based hit function: check if retrieved chunk contains info to answer the query.

    Args:
        doc: Retrieved document with page_content and metadata.
        case: Test case with "query" and "reference_answer".
        llm: LangChain LLM for evaluation (defaults to Config LLM via _build_llm).

    Returns:
        True if the chunk contains information needed to answer the query.
    """
    from app.dependencies import _build_llm

    _llm = llm or _build_llm()

    query = case.get("query", "")
    reference = case.get("reference_answer", "")
    chunk_text = doc.page_content

    prompt = f"""Given a query and a reference answer, determine if the provided text chunk contains sufficient information to answer the query.

Query: {query}

Reference Answer: {reference}

Text Chunk:
{chunk_text}

Does this chunk contain information that would help answer the query? Consider:
1. Does it contain key facts mentioned in the reference answer?
2. Does it provide relevant context for answering the query?
3. Could someone use this chunk to formulate a correct answer?

Answer only "yes" or "no"."""

    response = _llm.invoke(prompt)
    text = _message_text(response.content).strip().lower()
    return "yes" in text


def _doc_matches_relevant(doc: Document, relevant_set: set[str]) -> bool:
    """金标准为父块 id 时，子块命中也算 relevant。"""
    cid = str(doc.metadata.get("chunk_id", ""))
    if cid in relevant_set:
        return True
    parent = doc.metadata.get("chunk_parent_id")
    if parent and str(parent) in relevant_set:
        return True
    return False


def calculate_metrics_from_docs(docs: list[Document], relevant_ids: list[str], k: int) -> dict:
    """Recall@k / MRR 等，支持 parent chunk_id 与 child chunk_id 对齐。"""
    if not relevant_ids:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ap": 0.0}

    relevant_set = set(relevant_ids)
    top = docs[:k]
    hits = [_doc_matches_relevant(d, relevant_set) for d in top]

    matched_relevant = set()
    for d in top:
        cid = str(d.metadata.get("chunk_id", ""))
        parent = str(d.metadata.get("chunk_parent_id") or "")
        if cid in relevant_set:
            matched_relevant.add(cid)
        if parent in relevant_set:
            matched_relevant.add(parent)

    recall = len(matched_relevant) / len(relevant_set)
    precision = sum(hits) / k if k > 0 else 0.0

    mrr = 0.0
    for i, h in enumerate(hits, 1):
        if h:
            mrr = 1.0 / i
            break

    ap, running = 0.0, 0
    for i, h in enumerate(hits, 1):
        if h:
            running += 1
            ap += running / i
    ap = ap / len(relevant_set) if relevant_set else 0.0

    return {"recall": recall, "precision": precision, "mrr": mrr, "ap": ap}


def calculate_metrics(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> dict:
    """Recall@k, Precision@k, MRR, MAP from chunk_id lists."""
    if not relevant_ids:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ap": 0.0}

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_set & relevant_set)

    recall = hits / len(relevant_set)
    precision = hits / k if k > 0 else 0.0

    mrr = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            mrr = 1.0 / i
            break

    ap, hits_at_k = 0.0, 0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            hits_at_k += 1
            ap += hits_at_k / i
    ap = ap / len(relevant_set) if relevant_set else 0.0

    return {"recall": recall, "precision": precision, "mrr": mrr, "ap": ap}


def calculate_metrics_from_hits(hits: list[bool], num_relevant: int, k: int) -> dict:
    """Recall@k, Precision@k, MRR, MAP from a boolean hit list."""
    if num_relevant == 0:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ap": 0.0}

    hits_k = hits[:k]
    total_hits = sum(hits_k)

    recall = min(total_hits / num_relevant, 1.0)
    precision = total_hits / k if k > 0 else 0.0

    mrr = 0.0
    for i, h in enumerate(hits_k, 1):
        if h:
            mrr = 1.0 / i
            break

    ap, running = 0.0, 0
    for i, h in enumerate(hits_k, 1):
        if h:
            running += 1
            ap += running / i
    ap = ap / num_relevant

    return {"recall": recall, "precision": precision, "mrr": mrr, "ap": ap}


def evaluate_retrieval(
    retriever: Retriever,
    test_cases: list[dict],
    k: int = 5,
    fetch_k: int = 20,
    hit_fn: HitFn | None = None,
    llm: ChatOpenAI | None = None,
    verbose: bool = False,
) -> dict:
    """Evaluate retrieval on test cases.

    Args:
        retriever: Retriever instance.
        test_cases: List of dicts. Must contain "query" and either:
            - "relevant_ids": list[str]  for chunk_id matching
            - any fields consumed by hit_fn
        k: Top-k for metrics.
        fetch_k: Candidates before reranking.
        hit_fn: Optional callable(doc: Document, case: dict) -> bool.
                When provided, used instead of chunk_id matching.
        llm: LangChain LLM for is_hit_answer (ignored by other hit_fn).
        verbose: Print per-query details.

    Returns:
        Dict with averaged Recall@k, Precision@k, MRR, MAP.
    """
    all_metrics = []

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        results: list[Document] = retriever.retrieve(
            query, k=k, fetch_k=fetch_k, rerank=True, expand_parent=True
        )

        if hit_fn is not None:
            if hit_fn is is_hit_answer:
                hits = [is_hit_answer(doc, case, llm) for doc in results]
            else:
                hits = [hit_fn(doc, case) for doc in results]
            if "relevant_pages" in case or "relevant_layouts" in case:
                num_relevant = len(case.get("relevant_pages", case.get("relevant_layouts", [])))
                num_relevant = max(num_relevant, 1)
            else:
                # For answer-based evaluation without ground truth count,
                # assume we need at least k/2 relevant docs (or min 2)
                num_relevant = max(k // 2, 2)
            metrics = calculate_metrics_from_hits(hits, num_relevant, k)
        else:
            relevant_ids = case.get("relevant_ids", [])
            if not relevant_ids:
                continue
            retrieved_ids = [doc.metadata.get("chunk_id", "") for doc in results]
            metrics = calculate_metrics(retrieved_ids, relevant_ids, k)

        all_metrics.append(metrics)

        if verbose:
            print(f"\n[{i}] {query[:80]}")
            if hit_fn is not None:
                print(f"  hits={hits[:k]}  recall={metrics['recall']:.2%}  mrr={metrics['mrr']:.4f}")
            else:
                print(f"  recall={metrics['recall']:.2%}  precision={metrics['precision']:.2%}")

    n = len(all_metrics)
    if n == 0:
        return {"recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0, "map": 0.0, "num_queries": 0}

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        f"recall@{k}": _avg([m["recall"] for m in all_metrics]),
        f"precision@{k}": _avg([m["precision"] for m in all_metrics]),
        "mrr": _avg([m["mrr"] for m in all_metrics]),
        "map": _avg([m["ap"] for m in all_metrics]),
        "num_queries": n,
    }

