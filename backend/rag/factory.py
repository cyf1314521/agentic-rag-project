"""
RAG 组件工厂：Embedding、Reranker、Milvus Store、VLM 的单例创建。

避免重复加载 HuggingFace / CrossEncoder 模型；统一 hybrid collection 配置。
"""

import base64
from pathlib import Path
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from pymilvus import connections
from sentence_transformers import CrossEncoder


def ensure_milvus_orm_connection(store: Milvus) -> None:
    """
    langchain-milvus 用 MilvusClient 建连，但 add_documents 内部仍走 ORM Collection API。
    pymilvus 2.6+ 不会自动注册 ORM alias，需显式 connect，否则会 ConnectionNotExistException。
    """
    alias = store.alias
    if not connections.has_connection(alias):
        connections.connect(alias=alias, **store._connection_args)


class EmbeddingService:
    """按 model_name 缓存 HuggingFaceEmbeddings 实例。"""

    _instances = {}

    @classmethod
    def get_embeddings(cls, model_name: str) -> HuggingFaceEmbeddings:
        if model_name not in cls._instances:
            cls._instances[model_name] = HuggingFaceEmbeddings(model_name=model_name)
        return cls._instances[model_name]


class RerankerService:
    """按 model_name 缓存 CrossEncoder 实例。"""

    _instances = {}

    @classmethod
    def get_reranker(cls, model_name: str) -> CrossEncoder:
        if model_name not in cls._instances:
            cls._instances[model_name] = CrossEncoder(model_name)
        return cls._instances[model_name]


class MilvusStoreFactory:
    """创建带 BM25 稀疏向量 + 稠密向量的 Milvus 向量库封装。"""

    @staticmethod
    def create_store(
        embeddings: HuggingFaceEmbeddings,
        milvus_uri: str,
        collection_name: str,
        is_child: bool = True,
    ) -> Milvus:
        bm25 = BM25BuiltInFunction(input_field_names="text", output_field_names="sparse")
        suffix = "children" if is_child else "parents"

        store = Milvus(
            embeddings,
            builtin_function=bm25,
            vector_field=["dense", "sparse"],
            collection_name=f"{collection_name}_{suffix}",
            connection_args={"uri": milvus_uri},
        )
        ensure_milvus_orm_connection(store)
        return store


class VisionService:
    """
    视觉语言模型服务：将论文图表 PNG 以 base64 多模态消息发给 VLM。

    分析结果缓存到 citation metadata 的 vlm_description 字段。
    """

    _instance = None

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @classmethod
    def get_instance(cls, llm: BaseChatModel = None):
        if cls._instance is None and llm:
            cls._instance = cls(llm)
        return cls._instance

    def analyze_figure(self, image_path: str, caption: str = "") -> str:
        """返回图表的文字描述，失败返回空字符串。"""
        if not Path(image_path).exists():
            return ""

        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""

        prompt = (
            "Analyze this figure from an academic paper. Describe:\n"
            "1. Chart/diagram type (bar chart, line plot, architecture diagram, etc.)\n"
            "2. Key visual elements (axes, labels, trends, components)\n"
            "3. Main findings or patterns shown\n"
            "4. Numerical values if visible\n\n"
            "Be concise and focus on information useful for answering research questions."
        )
        if caption:
            prompt += f"\n\nCaption context: {caption}"

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ]
        )

        try:
            response = self.llm.invoke([message])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"VLM analysis failed: {e}")
            return ""


def is_visual_query(query: str) -> bool:
    """启发式判断：用户问题是否明确涉及图表/可视化。"""
    visual_keywords = [
        "show", "display", "visualize", "plot", "chart", "graph", "diagram",
        "figure", "illustration", "image", "picture", "看图", "图中", "图表",
        "what does", "what is shown", "describe the figure", "describe the image",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in visual_keywords)


def should_invoke_vlm(query: str, has_figure: bool, answer: str = "") -> bool:
    """
    决定是否调用 VLM：
    1. 有图且为视觉类问题 → 主动调用（generate 阶段）
    2. 有图且文本答案明确表示信息不足 → 兜底调用（reflect 阶段）
    """
    if not has_figure:
        return False

    if is_visual_query(query):
        return True

    if answer:
        insufficient_indicators = [
            "not contain sufficient information", "insufficient",
            "cannot answer", "no relevant information",
            "信息不足", "无法回答",
        ]
        if any(ind in answer.lower() for ind in insufficient_indicators):
            return True

    return False
