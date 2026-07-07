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
    # 嵌入/重排设备：auto（有 CUDA 则用 GPU）| cuda | cpu
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
    # 已缓存模型时仅读本地，避免启动反复访问 huggingface.co 超时（WinError 10060）
    HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "true").lower() == "true"

    # ---------- 检索超参数 ----------
    FETCH_K = int(os.getenv("FETCH_K", "20"))   # 重排前从向量库拉取的候选条数
    TOP_K = int(os.getenv("TOP_K", "5"))       # 最终返回给 LLM 的文档条数
    RRF_K = int(os.getenv("RRF_K", "60"))      # BM25 与稠密向量 RRF 融合的常数 k

    # ---------- 检索结果缓存 ----------
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

    # ---------- 聊天链路可观测（RAG 召回 + Agent 各阶段）----------
    CHAT_TRACE = os.getenv("CHAT_TRACE", "true").lower() == "true"
    CHAT_TRACE_SAVE = os.getenv("CHAT_TRACE_SAVE", "true").lower() == "true"
    CHAT_TRACE_DIR = os.getenv("CHAT_TRACE_DIR", "./data/traces")
    CHAT_TRACE_PREVIEW_CHARS = int(os.getenv("CHAT_TRACE_PREVIEW_CHARS", "280"))

    # ---------- 主 LLM（OpenAI 兼容 API：Ollama / vLLM 等）----------
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))  # 单次 LLM 请求超时（秒），Ollama 本地模型宜设大一些
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # 子 Agent 反思不足时的最大重试次数

    # ---------- Agent 容错：子图超时（秒）----------
    # 单路子图（retrieve→generate→reflect 含重试）的总预算；须 < REQUEST_TIMEOUT
    SUB_AGENT_TIMEOUT = int(os.getenv("SUB_AGENT_TIMEOUT", "90"))

    # ---------- Agent 容错：整请求 deadline + 单步超时（秒）----------
    # 须 > SUB_AGENT_TIMEOUT + summarize/intent/analyze 余量（金字塔顶层）
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
    EVAL_REQUEST_TIMEOUT = int(os.getenv("EVAL_REQUEST_TIMEOUT", "300"))
    RETRIEVE_STEP_TIMEOUT = int(os.getenv("RETRIEVE_STEP_TIMEOUT", "30"))
    GENERATE_STEP_TIMEOUT = int(os.getenv("GENERATE_STEP_TIMEOUT", "25"))
    REFLECT_STEP_TIMEOUT = int(os.getenv("REFLECT_STEP_TIMEOUT", "8"))
    SYNTHESIZE_STEP_TIMEOUT = int(os.getenv("SYNTHESIZE_STEP_TIMEOUT", "30"))

    # ---------- Agent 容错：有界重试 ----------
    RETRIEVE_MAX_RETRIES = int(os.getenv("RETRIEVE_MAX_RETRIES", "3"))
    RETRIEVE_RETRY_BACKOFF_MS = os.getenv("RETRIEVE_RETRY_BACKOFF_MS", "200,500,1000")
    LLM_NODE_MAX_RETRIES = int(os.getenv("LLM_NODE_MAX_RETRIES", "2"))
    LLM_NODE_RETRY_BACKOFF_MS = os.getenv("LLM_NODE_RETRY_BACKOFF_MS", "1000,3000")

    # ---------- 视觉语言模型 VLM（图表理解，可选）----------
    VLM_ENABLED = os.getenv("VLM_ENABLED", "false").lower() == "true"
    VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:11434/v1")
    VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl")
    VLM_API_KEY = os.getenv("VLM_API_KEY", "ollama")

    # ---------- PostgreSQL：会话元数据 + LangGraph checkpoint ----------
    POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/scholar_rag")

    # ---------- PDF 解析（Docling）----------
    # true：降低 batch、关闭可选视觉管线，减轻 std::bad_alloc（大 PDF / 内存紧张时建议开启）
    DOCLING_LOW_MEMORY = os.getenv("DOCLING_LOW_MEMORY", "true").lower() == "true"
    # true：每次上传将解析全文 + 阶段明细写入 PARSE_ARTIFACT_DIR
    SAVE_PARSE_ARTIFACT = os.getenv("SAVE_PARSE_ARTIFACT", "true").lower() == "true"
    PARSE_ARTIFACT_DIR = os.getenv("PARSE_ARTIFACT_DIR", "./data/parsed")

    # ---------- 文件上传 ----------
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    # ---------- 论文画像（入库压缩，供主题/类型检索）----------
    PAPER_PROFILE_ENABLED = os.getenv("PAPER_PROFILE_ENABLED", "true").lower() == "true"
    PAPER_PROFILE_MAX_SOURCE_CHARS = int(os.getenv("PAPER_PROFILE_MAX_SOURCE_CHARS", "12000"))
    PROFILE_DISCOVERY_TOP_K = int(os.getenv("PROFILE_DISCOVERY_TOP_K", "5"))

    # ---------- Intent 解析 ----------
    INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.75"))
    # 检索后置信度门：CrossEncoder 重排 top1 低于此值视为「与论文关联度低」
    RETRIEVAL_MIN_RERANK_SCORE = float(os.getenv("RETRIEVAL_MIN_RERANK_SCORE", "0.15"))

    # ---------- Corpus Gate（前置网关，不查 Milvus）----------
    CONTENT_GATE_ENABLED = os.getenv("CONTENT_GATE_ENABLED", "true").lower() == "true"
    CONTENT_GATE_MODE = os.getenv("CONTENT_GATE_MODE", "hybrid")  # hybrid | embedding | bm25
    CONTENT_GATE_MIN_SCORE = float(os.getenv("CONTENT_GATE_MIN_SCORE", "0.15"))
    CONTENT_GATE_HIGH_SCORE = float(os.getenv("CONTENT_GATE_HIGH_SCORE", "0.25"))
    CONTENT_GATE_USE_PREV_TURN = os.getenv("CONTENT_GATE_USE_PREV_TURN", "true").lower() == "true"
    CONTENT_GATE_EMB_WEIGHT = float(os.getenv("CONTENT_GATE_EMB_WEIGHT", "0.75"))
    CONTENT_GATE_BM25_WEIGHT = float(os.getenv("CONTENT_GATE_BM25_WEIGHT", "0.25"))
    TASK_OFF_DOMAIN_ENABLED = os.getenv("TASK_OFF_DOMAIN_ENABLED", "true").lower() == "true"
    TASK_OVERRIDE_MIN_SCORE = float(os.getenv("TASK_OVERRIDE_MIN_SCORE", "0.35"))

    # ---------- Evidence gate ----------
    RETRIEVAL_MIN_FALLBACK_SCORE = float(os.getenv("RETRIEVAL_MIN_FALLBACK_SCORE", "0.08"))

    # ---------- Slot inheritance ----------
    SLOT_INHERITANCE_ENABLED = os.getenv("SLOT_INHERITANCE_ENABLED", "true").lower() == "true"

    # ---------- Compliance gate ----------
    COMPLIANCE_GATE_ENABLED = os.getenv("COMPLIANCE_GATE_ENABLED", "false").lower() == "true"
    COMPLIANCE_DENYLIST_PATH = os.getenv("COMPLIANCE_DENYLIST_PATH", "./data/compliance/denylist.txt")

    # ---------- Grounding check ----------
    GROUNDING_CHECK_ENABLED = os.getenv("GROUNDING_CHECK_ENABLED", "true").lower() == "true"
    GROUNDING_MIN_CITATION_RATIO = float(os.getenv("GROUNDING_MIN_CITATION_RATIO", "0.2"))

    # ---------- HTTP 服务监听 ----------
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
