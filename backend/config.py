"""
应用配置模块：从环境变量（.env）加载全部运行参数。

所有配置项均可通过 backend/.env 覆盖，便于本地开发与 Docker 部署时分别设置。
"""

import os
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
load_dotenv()


class Config:
    """集中管理 ScholarRAG 后端所需的全部配置常量。"""

    # ---------- Milvus 向量数据库 ----------
    MILVUS_URI = os.getenv("MILVUS_URI", "./data/milvus.db")  # 连接地址，Docker 内通常为 http://milvus:19530
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers")  # 集合名前缀，实际会派生 _children / _parents

    # ---------- 嵌入与重排序模型（HuggingFace 路径或模型 ID）----------
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    # 已缓存模型时仅读本地，避免启动反复访问 huggingface.co 超时（WinError 10060）
    HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "true").lower() == "true"

    # ---------- 检索超参数 ----------
    FETCH_K = int(os.getenv("FETCH_K", "20"))   # 重排前从向量库拉取的候选条数
    TOP_K = int(os.getenv("TOP_K", "5"))       # 最终返回给 LLM 的文档条数
    RRF_K = int(os.getenv("RRF_K", "60"))      # BM25 与稠密向量 RRF 融合的常数 k

    # ---------- 检索结果缓存 ----------
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

    # ---------- 主 LLM（OpenAI 兼容 API：Ollama / vLLM 等）----------
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:32b")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))  # 单次 LLM 请求超时（秒），Ollama 本地模型宜设大一些
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # 子 Agent 反思不足时的最大重试次数

    # ---------- 视觉语言模型 VLM（图表理解，可选）----------
    VLM_ENABLED = os.getenv("VLM_ENABLED", "false").lower() == "true"
    VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")
    VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl")
    VLM_API_KEY = os.getenv("VLM_API_KEY", "ollama")

    # ---------- PostgreSQL：会话元数据 + LangGraph checkpoint ----------
    POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/scholar_rag")

    # ---------- 文件上传 ----------
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    # ---------- HTTP 服务监听 ----------
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
