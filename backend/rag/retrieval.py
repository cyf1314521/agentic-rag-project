"""
混合检索模块：BM25 + 稠密向量 RRF 融合 → CrossEncoder 重排 → 父块回溯。

可选 HyDE 查询扩展；支持 node_type / section_type 元数据过滤；
结果可经 RetrievalCache 做语义相似 query 缓存。
"""

import logging  # 标准库：用于记录检索过程中的日志（警告、调试、信息等）
from typing import Optional  # 类型标注：表示参数或返回值可以是某类型或 None
from langchain_core.language_models import BaseChatModel

from app.llm_utils import message_content_to_str
from langchain_core.documents import Document  # LangChain 文档对象，含 page_content 文本与 metadata 元数据字典
from langchain_milvus import Milvus  # LangChain 对 Milvus 向量库的封装，提供 similarity_search 等接口
from pymilvus import Function, FunctionType  # Milvus 原生：定义混合检索中的 RRF 融合函数及其类型枚举
from .cache import RetrievalCache  # 同包模块：按查询语义相似度缓存检索结果，避免重复计算
from .factory import EmbeddingService, RerankerService, MilvusStoreFactory  # 工厂类：统一创建嵌入模型、重排模型、Milvus 存储实例

logger = logging.getLogger(__name__)  # 以当前模块名创建 logger，日志中会显示 rag.retrieval 便于定位

# 重排前多取候选倍数；去重前父块扩展倍数
RERANK_FETCH_MULTIPLIER = 2  # 开启 CrossEncoder 重排时，混合检索先取 fetch_k×2 条，再重排截到 fetch_k
DEDUP_FETCH_MULTIPLIER = 2  # 父块扩展或截断时先保留 k×2 条，去重后再取最终 k 条，降低去重后数量不足的风险


class Retriever:
    """
    混合检索器：在 child collection 上检索，可选扩展为 parent chunk。

    典型调用链（retrieve 方法）：
    缓存 → HyDE → 混合搜索 → 重排 → 父块扩展 → chunk_id 去重 → Top-K
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",  # 稠密向量嵌入模型名称，用于 query/doc 向量化
        reranker_model: str = "BAAI/bge-reranker-v2-m3",  # CrossEncoder 重排模型名称
        milvus_uri: str = "http://localhost:19530",  # Milvus 服务地址
        collection_name: str = "papers",  # 逻辑集合名，工厂会据此区分 child/parent 物理 collection
        llm: Optional[BaseChatModel] = None,  # 大语言模型实例；为 None 时无法使用 HyDE
        enable_cache: bool = True,  # 是否启用检索结果缓存
        load_reranker: bool = True,  # False 时跳过重排模型加载（评测/低内存）
        child_store: Optional[Milvus] = None,  # 可注入已建好的子块 Milvus 实例，便于测试或复用连接
        parent_store: Optional[Milvus] = None,  # 可注入已建好的父块 Milvus 实例
    ):
        # 通过单例工厂获取嵌入模型，避免重复加载权重
        self.embeddings = EmbeddingService.get_embeddings(embedding_model)
        # 通过工厂获取重排模型（CrossEncoder）；load_reranker=False 时 retrieve(rerank=False) 仍可用
        self.reranker = (
            RerankerService.get_reranker(reranker_model) if load_reranker else None
        )
        self.milvus_uri = milvus_uri  # 保存 URI，供 get_updater 等后续逻辑使用
        self.collection_name = collection_name  # 保存集合名
        self.llm = llm  # 保存 LLM 引用，HyDE 时调用 invoke
        # 启用缓存则构造 RetrievalCache（内部用 embeddings 算 query 向量做相似匹配）；否则为 None
        self.cache = RetrievalCache(self.embeddings) if enable_cache else None

        # 子块向量库：检索主战场；未传入则按默认参数创建 child collection 连接
        self._child_store = child_store or MilvusStoreFactory.create_store(
            self.embeddings, milvus_uri, collection_name, is_child=True
        )
        # 父块向量库：子块命中后按 parent_id 回溯完整段落；is_child=False 指向 parent collection
        self._parent_store = parent_store or MilvusStoreFactory.create_store(
            self.embeddings, milvus_uri, collection_name, is_child=False
        )

    def retrieve(
        self,
        query: str,  # 用户原始查询文本
        k: int = 5,  # 最终返回的文档条数 Top-K
        use_hyde: bool = False,  # 是否先用 LLM 生成假设性段落再检索（HyDE）
        rerank: bool = True,  # 是否用 CrossEncoder 对混合检索候选重排
        expand_parent: bool = True,  # 是否将子块结果扩展为父块完整语义单元
        rrf_k: int = 60,  # Milvus RRF 融合公式中的常数 k，越大则排名靠后项衰减越慢
        fetch_k: int = 20,  # 混合检索（及重排后）保留的候选数量，再经扩展/去重得到 k
        node_type_filter: Optional[list[str]] = None,  # 元数据过滤：仅保留指定 node_type 的块
        section_type_filter: Optional[list[str]] = None,  # 元数据过滤：仅保留指定 section_type 的块
        paper_id_filter: Optional[list[str]] = None,  # 会话 scope：仅检索这些 paper_id
    ) -> list[Document]:
        """执行完整检索管道，返回 Top-K 个 Document。"""
        # 集合被 manage 路由 drop 后需清空 langchain-milvus 内部缓存，否则仍指向已删除的 collection
        self._child_store._col_cache = None  # 清空子库 Milvus 客户端缓存的 Collection 对象
        self._child_store._cache_key = None  # 清空子库用于判断缓存是否有效的 key
        self._parent_store._col_cache = None  # 清空父库 Collection 缓存
        self._parent_store._cache_key = None  # 清空父库 cache key

        if self._child_store.col is None:  # col 为 None 表示 Milvus 中尚不存在对应 collection（未建索引）
            logger.warning("Collection not found, no documents indexed yet.")  # 打警告日志
            return []  # 无数据可检，直接返回空列表

        # 带 scope / 章节过滤时不走缓存（缓存键仅含 query，会串库到其它 paper_id）
        use_cache = (
            self.cache is not None
            and not paper_id_filter
            and not node_type_filter
            and not section_type_filter
        )
        if use_cache:
            cache = self.cache
            assert cache is not None
            cached = cache.get(query)
            if cached is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached

        # 若开启 HyDE 且配置了 llm，则用假设性段落作为检索 query；否则仍用原始 query
        search_query = self._hyde(query) if use_hyde and self.llm else query
        # 根据 node_type / section_type 列表拼 Milvus 布尔过滤表达式；无过滤则为 None
        expr = self._build_expr(node_type_filter, section_type_filter, paper_id_filter)

        if rerank and self.reranker:  # 需要重排且重排模型可用
            # 混合检索多取 fetch_k×2 条候选，供 CrossEncoder 筛选
            children = self._hybrid_search(
                self._child_store, search_query, fetch_k * RERANK_FETCH_MULTIPLIER, rrf_k, expr
            )
            if not children:  # 混合检索零结果
                logger.warning(f"No results found for query: {query[:50]}...")
                return []
            # 用原始 query（非 HyDE 文本）与 passage 配对打分，保留 top fetch_k
            children = self._rerank(query, children, fetch_k)
        else:  # 不重排：混合检索直接取 fetch_k 条
            children = self._hybrid_search(self._child_store, search_query, fetch_k, rrf_k, expr)
            if not children:
                logger.warning(f"No results found for query: {query[:50]}...")
                return []

        if expand_parent:  # 需要回溯父块
            # 先取前 k×2 个子块结果，再按 chunk_parent_id 拉父 collection
            results = self._expand_to_parents(children[: k * DEDUP_FETCH_MULTIPLIER])
        else:  # 不扩展父块，直接使用子块（仍多取 k×2 条以便去重后够 k 条）
            results = children[: k * DEDUP_FETCH_MULTIPLIER]

        seen = set()  # 记录已出现的 chunk_id，用于去重
        deduped = []  # 去重后的 Document 列表
        for doc in results:  # 按检索/扩展后的顺序遍历
            # 优先用 metadata 中的 chunk_id；缺失则用 Python 对象 id 作为唯一键（兜底）
            cid = doc.metadata.get("chunk_id", id(doc))
            if cid not in seen:  # 第一次见到该 chunk_id
                seen.add(cid)  # 标记为已见
                deduped.append(doc)  # 保留该文档

        final = deduped[:k]  # 去重后截取前 k 条作为最终结果

        if use_cache:
            cache = self.cache
            assert cache is not None
            cache.put(query, final)

        logger.info(f"Retrieved {len(final)} results for query: {query[:50]}...")  # 记录最终条数
        return final  # 返回 Top-K Document 列表

    def get_updater(self):
        """返回 IncrementalUpdater，用于按 paper_id 删改向量。"""
        from .incremental import IncrementalUpdater  # 延迟导入，避免 retrieval ↔ incremental 循环依赖
        # 传入父/子 Milvus 存储，增量更新时可同时维护两套 collection
        return IncrementalUpdater(self._parent_store, self._child_store)

    def _build_expr(
        self,
        node_type_filter: Optional[list[str]],
        section_type_filter: Optional[list[str]],
        paper_id_filter: Optional[list[str]] = None,
    ) -> Optional[str]:
        """构造 Milvus 布尔过滤表达式。"""
        parts = []  # 存放各组过滤条件字符串，最后用 && 连接
        if paper_id_filter:
            safe = [pid.replace("\\", "\\\\").replace('"', '\\"') for pid in paper_id_filter]
            ids = ", ".join(f'"{pid}"' for pid in safe)
            parts.append(f"(paper_id in [{ids}])")
        if node_type_filter:  # 若指定了 node_type 白名单
            # 将 ["a","b"] 拼成 (node_type == "a" || node_type == "b")，同一字段多值为 OR
            types = " || ".join(f'node_type == "{t}"' for t in node_type_filter)
            parts.append(f"({types})")  # 外层括号保证与 section 条件组合时优先级清晰
        if section_type_filter:  # 若指定了 section_type 白名单
            types = " || ".join(f'section_type == "{t}"' for t in section_type_filter)
            parts.append(f"({types})")
        # 两组都有则用 && 表示同时满足；无任何过滤则返回 None（检索时不带 expr）
        return " && ".join(parts) if parts else None

    def _hybrid_search(
        self,
        store: Milvus,  # 在哪个 Milvus 封装实例上搜（通常为 _child_store）
        query: str,  # 用于生成 dense 向量并触发 sparse(BM25) 的查询文本
        k: int,  # 最终返回条数
        rrf_k: int,  # RRF 融合参数
        expr: Optional[str] = None,  # 可选的 Milvus 标量过滤表达式
    ) -> list[Document]:
        """dense + sparse(BM25) 经 Milvus 内置 RRF 融合。"""
        # 定义 Milvus 2.x 的 Rerank 函数：对 dense、sparse 两路召回做 Reciprocal Rank Fusion
        reranker = Function(
            name="rrf_reranker",  # 函数在检索请求中的逻辑名称
            function_type=FunctionType.RERANK,  # 类型为重排/融合，而非标量运算等
            input_field_names=["dense", "sparse"],  # 参与融合的两个向量字段名（需与 schema 一致）
            params={"k": rrf_k},  # 传入 RRF 公式中的 k 常数
        )
        # similarity_search 参数：返回 k 条、使用上述 reranker、每路先 fetch_k 条再融合
        kwargs = {"k": k, "reranker": reranker, "fetch_k": k}
        if expr:  # 有过滤条件时加入 expr，Milvus 先过滤再检索
            kwargs["expr"] = expr
        # LangChain Milvus 封装：对 query 做嵌入 + BM25，执行混合检索并 RRF
        results = store.similarity_search(query, **kwargs)
        return results  # Document 列表，metadata 含 chunk_id、chunk_parent_id 等

    def _rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """CrossEncoder 对 (query, passage) 打分后取 top_k。"""
        if not docs or self.reranker is None:
            return docs[:top_k]
        # 构造 (查询, 段落正文) 对，供 CrossEncoder 联合编码打分
        pairs = [(query, doc.page_content) for doc in docs]
        raw_scores = self.reranker.predict(pairs)  # ndarray，CrossEncoder 相关性分数
        score_list = [float(s) for s in raw_scores]  # 转为 list[float]，便于 zip + sorted 通过类型检查
        ranked = sorted(
            zip(docs, score_list),
            key=lambda item: item[1],
            reverse=True,
        )
        return [doc for doc, _ in ranked[:top_k]]

    def _expand_to_parents(self, children: list[Document]) -> list[Document]:
        """子块命中后按 chunk_parent_id 拉取父 collection 中的完整语义单元。"""
        # dict.fromkeys 去重且保持首次出现顺序；只收集有 chunk_parent_id 的子块
        parent_ids = list(
            dict.fromkeys(
                doc.metadata.get("chunk_parent_id")
                for doc in children
                if doc.metadata.get("chunk_parent_id")
            )
        )

        if not parent_ids:  # 子块元数据里没有父 id，无法扩展
            return children  # 原样返回子块列表

        parents = []  # 从父 collection 查到的 Document
        for pid in parent_ids:  # 逐个父 chunk_id 查询
            try:
                expr = f'chunk_id == "{pid}"'  # 在父库中用标量字段 chunk_id 精确匹配
                # 查询文本用 "dummy" 即可：主要靠 expr 过滤，向量相似度在此不重要
                hits = self._parent_store.similarity_search("dummy", k=1, expr=expr)
                parents.extend(hits)  # 将命中的父块追加到列表
            except Exception as e:  # 单条父块查询失败不中断整个流程
                logger.error(f"Failed to fetch parent {pid}: {e}")

        # 若至少查到一条父块则返回父块列表；否则降级返回原子块 children
        return parents if parents else children

    def _hyde(self, query: str) -> str:
        """Hypothetical Document Embeddings：用 LLM 生成假设性段落再检索。"""
        # 英文 prompt：要求写一段能回答问题的论文节选，不要解释
        prompt = (
            "Please write a short passage from an academic paper that would answer "
            f"the following question. Do not explain, just write the passage.\n\n"
            f"Question: {query}\n\nPassage:"
        )
        if not self.llm:
            return query
        try:
            response = self.llm.invoke(prompt)  # 调用 LLM 生成文本
            content = (
                message_content_to_str(response.content)
                if hasattr(response, "content")
                else str(response)
            )
            return content.strip()  # 去掉首尾空白作为 HyDE 检索 query
        except Exception:
            return query  # LLM 失败则回退为原始 query，保证检索仍可继续
