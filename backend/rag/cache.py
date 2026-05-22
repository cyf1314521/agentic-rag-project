"""
检索结果缓存：基于查询向量相似度的 LRU 缓存。

与精确字符串匹配不同：对语义相近的 query（余弦相似度 >= threshold）
直接返回历史检索结果，减少重复的 Milvus + 重排开销。
"""

import logging
import threading
from collections import OrderedDict
from typing import Optional
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算两向量的余弦相似度。"""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class RetrievalCache:
    """
    LRU 缓存：键为原始 query 字符串，值为 (query_embedding, Document 列表)。

    get：嵌入当前 query，在缓存中找最相似条目，超过阈值则命中。
    put：写入新结果，超 max_size 时淘汰最久未使用项。
    """

    def __init__(
        self,
        embeddings,
        max_size: int = 1000,
        similarity_threshold: float = 0.95,
    ):
        self.embeddings = embeddings
        self.max_size = max_size
        self.threshold = similarity_threshold
        self._store: OrderedDict[str, tuple[np.ndarray, list[Document]]] = OrderedDict()
        # 并行 sub_agent 会同时 retrieve；保护 OrderedDict 避免迭代时被 put 修改
        self._lock = threading.Lock()

    def _embed(self, query: str) -> np.ndarray:
        vec = self.embeddings.embed_query(query)
        return np.array(vec, dtype=np.float32)

    def _find_best(self, vec: np.ndarray) -> Optional[tuple[str, float]]:
        """返回 (cache_key, similarity) 或 None。"""
        best_key, best_sim = None, -1.0
        for key, (cached_vec, _) in self._store.items():
            sim = _cosine(vec, cached_vec)
            if sim > best_sim:
                best_sim, best_key = sim, key
        if best_key is not None and best_sim >= self.threshold:
            return best_key, best_sim
        return None

    def get(self, query: str) -> Optional[list[Document]]:
        vec = self._embed(query)
        with self._lock:
            if not self._store:
                return None
            match = self._find_best(vec)
            if match:
                key, sim = match
                self._store.move_to_end(key)
                logger.debug(f"Cache hit (sim={sim:.3f}): {query[:60]}")
                return self._store[key][1]
        return None

    def put(self, query: str, results: list[Document]) -> None:
        vec = self._embed(query)
        with self._lock:
            if query in self._store:
                self._store.move_to_end(query)
            self._store[query] = (vec, results)
            if len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
