
---

<p align="center">
  <img src="doc/logo.png" alt="logo" width="720">
</p>

<div align="center">

# ScholarRAG

**Multi-Agent RAG System for Academic Paper Q&A**

Upload academic papers, ask questions in natural language, get grounded answers with precise citations.

![Python](https://img.shields.io/badge/python-3.12+-blue)
![React](https://img.shields.io/badge/react-18-61dafb)
![LangGraph](https://img.shields.io/badge/LangGraph-0.x-orange)
![Milvus](https://img.shields.io/badge/Milvus-2.5+-00bfa5)
![License](https://img.shields.io/badge/license-MIT-green)

[Quick Start](#-quick-start) · [Features](#-features) · [Architecture](#-architecture) · [Evaluation](#-evaluation) · [API Reference](#-api-reference)

</div>

> [!NOTE]
> Still undergoing continuous optimization and updates.

## What is ScholarRAG?

https://github.com/user-attachments/assets/5f9d36e9-9027-4fcd-b0f4-b0dee7d123a3

ScholarRAG is an end-to-end academic paper Q&A system. It parses PDFs with full structural awareness (sections, tables, figures), retrieves relevant passages via hybrid search, and generates cited answers through a multi-agent pipeline — all accessible through a clean chat interface.

**Key highlights:**

- **Session-scoped retrieval** — each chat session binds uploaded papers; retrieval only searches papers in that session
- **Multi-agent pipeline** — query decomposition, parallel sub-agents with self-reflection, final synthesis with citation remapping
- **Hybrid retrieval** — BM25 + dense embedding fusion (RRF), cross-encoder reranking, parent-child chunk expansion
- **Structured PDF parsing** — Docling-based parsing with section hierarchy, table linearization, figure/caption linking
- **Multi-PDF focus synthesis** — when a session has multiple papers, citations are pooled and focus-paper inference narrows synthesis scope
- **Smart OCR fallback** — fast text extraction by default; OCR only when text density is too low
- **Query classification** — routes retrieval by section type (`experiment` / `method` / `background`); abstract queries bypass section filters
- **Multimodal figure understanding** — lazy VLM invocation for visual queries or insufficient text answers
- **Chat trace** — structured JSON logs for every request (classify, retrieve, reflect, synthesize)
- **Dual-layer evaluation** — retrieval metrics + end-to-end scope regression on a shared 20-case dataset

## Who is this for?

This project is **beginner-friendly** and well-suited for anyone looking to learn and practice the full Agentic RAG workflow — from PDF ingestion and hybrid retrieval to multi-agent orchestration with LangGraph. The codebase is modular, well-decoupled, and easy to follow, making it an ideal starting point for students and developers exploring RAG system design.

---

## Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Evaluation](#-evaluation)
- [Observability](#-observability)
- [Tech Stack](#-tech-stack)
- [Security Notice](#-security-notice)
- [License](#-license)
- [Contributors](#-contributors)

---

## Features

<p align="center">
  <img src="doc/demo.gif" alt="Demo" width="800">
</p>

| Category | Details |
|---|---|
| **Retrieval** | BM25 + dense fusion (RRF), cross-encoder reranking, parent-child chunk expansion, session `paper_id` filter |
| **PDF Parsing** | Docling with section hierarchy, table linearization, formula extraction, figure/caption linking |
| **Smart OCR** | Fast text extraction by default; auto-fallback to full OCR when text density is too low |
| **Figure Extraction** | bbox-based figure cropping saved per paper (PyMuPDF) |
| **Query Routing** | LLM classifies queries into `experimental_result` / `method` / `background` / `general`; abstract queries force `general` (no section filter) |
| **VLM Integration** | Lazy figure analysis for visual queries or when reflection finds insufficient text evidence |
| **Agent** | LangGraph: classify → decompose → parallel sub-agents → prepare_synthesis → synthesize |
| **Reflection** | Sub-agents self-evaluate sufficiency, retry with refined queries or trigger VLM fallback |
| **Multi-PDF synthesis** | Citation pooling + focus-paper inference; filters sub-answers and citations to the inferred target paper |
| **Memory** | Sliding window + LLM summary compression for multi-turn context |
| **Streaming** | SSE real-time streamed responses |
| **Citations** | Auto-generated source references (paper, section, page) with 1-based `[n]` indices |
| **Evaluation** | Scope regression (`run_scope_eval.py`); retrieval metrics (`run_retrieval_eval.py`) on shared dataset |
| **Observability** | `CHAT_TRACE` JSON under `data/traces/` for debugging retrieval and agent stages |

---

## Architecture

<div align="center">
  <img src="doc/scholsr_rag.png" alt="Architecture Diagram" width="720">
</div>

**Main graph:**

```
START → summarize → classify → analyze → [sub_agent × N] → prepare_synthesis → synthesize → END
```

**Sub-agent graph (one per sub-query, runs in parallel):**

```
START → retrieve → generate → reflect → (retry → prepare_retry → retrieve | END)
```

**Session scope:** Uploading PDFs binds them to the chat session. Every retrieval call passes `paper_id_filter` from the session's paper list — the same behavior the evaluation scripts simulate with `--compare` / `--single-paper`.

---

## Project Structure

```
scholar-rag/
├── backend/                          # Python backend (FastAPI + LangGraph + RAG)
│   ├── app/                          # FastAPI application layer
│   │   ├── main.py                   # Entry point: routes, CORS, mount frontend static files
│   │   ├── dependencies.py           # Singleton init: LLM, Retriever, PDFParser, PostgreSQL checkpointer
│   │   ├── store.py                  # SQLite session and file metadata storage
│   │   └── routers/
│   │       ├── chat.py               # POST /api/chat — SSE streaming conversation
│   │       ├── sessions.py           # Session list, history, delete
│   │       ├── files.py              # PDF upload (SHA256 dedup, Docling parse, Milvus indexing)
│   │       └── manage.py             # Collection clear, health check
│   │
│   ├── agent/                        # LangGraph multi-agent layer
│   │   ├── states.py                 # AgentState, SubAgentState, SubAnswer; custom reducers
│   │   ├── graph.py                  # Main + sub-graph assembly
│   │   ├── nodes.py                  # classify / analyze / retrieve / generate / reflect / synthesize
│   │   ├── prompts.py                # QUERY_CLASSIFIER, ANALYZER, SYNTHESIZER, GENERATOR, REFLECTOR, SUMMARIZER
│   │   ├── tools.py                  # SECTION_TYPE_ROUTE + ContextVar for query-type → section filter
│   │   └── checkpointer.py           # Memory / Postgres checkpointer factory
│   │
│   ├── rag/                          # RAG retrieval and parsing core
│   │   ├── models.py                 # PaperNode data model
│   │   ├── integration.py            # PDFParser, RAGIntegration (chunking, Milvus indexing)
│   │   ├── node_generator.py         # 6 node-type content generators (paragraph, table, figure, …)
│   │   ├── retrieval.py              # Hybrid retriever: BM25 + dense RRF, rerank, parent expand
│   │   ├── factory.py                # EmbeddingService, RerankerService, MilvusStoreFactory, VisionService
│   │   ├── citation.py               # CitationExtractor + scope helpers (parse indices, focus check)
│   │   ├── chat_trace.py             # Per-request trace: classify, retrieve, reflect, synthesize events
│   │   ├── cache.py                  # LRU retrieval cache
│   │   └── incremental.py            # Incremental Milvus updates by paper_id
│   │
│   ├── eval/                         # Evaluation system
│   │   ├── eval_retrieval.py         # Recall@k, Precision@k, MRR, MAP
│   │   ├── retrieval_eval_runner.py    # Shared runner for retrieval ablation / channel comparison
│   │   ├── fixtures/
│   │   │   ├── scope_eval_dataset.json   # 20 cases + must_contain + relevant_ids
│   │   │   └── SCOPE_EVAL_README.md
│   │   ├── RETRIEVAL_EVAL.md
│   │   └── results/                  # Latest eval JSON outputs
│   │
│   ├── scripts/
│   │   ├── run_scope_eval.py         # End-to-end agent scope regression
│   │   ├── run_retrieval_eval.py     # Retrieval-only metrics + ablation
│   │   ├── retrieval_eval_preview.py # Preview top-k hits for labeling relevant_ids
│   │   └── auto_label_relevant_ids.py
│   │
│   ├── test/                         # pytest tests
│   ├── data/
│   │   ├── traces/                   # CHAT_TRACE JSON (chat + eval runs)
│   │   ├── figures/                  # Extracted figure images
│   │   └── parsed/                   # Parse artifacts (when SAVE_PARSE_ARTIFACT=true)
│   ├── uploads/                      # Uploaded PDF originals
│   ├── config.py                     # All env-var configuration
│   ├── run.py                        # Windows-safe backend entry (SelectorEventLoop)
│   ├── requirements.txt
│   ├── requirements-gpu.txt          # CUDA torch for GPU embedding/reranker
│   ├── Dockerfile
│   └── .env.example
│
├── frontend/                         # React + Vite + TailwindCSS
│   ├── src/
│   │   ├── App.jsx                   # Main layout, SSE streaming, session switching
│   │   ├── api.js                    # fetch + ReadableStream SSE client
│   │   └── components/               # Sidebar, ChatMessages, ChatInput, FileUpload, SettingsPanel
│   ├── Dockerfile
│   └── nginx.conf
│
├── doc/                              # Logo, architecture diagram, demo GIF
├── docker-compose.yml                # backend + frontend + milvus + postgres
├── Makefile                          # install / dev / build / docker-up / test
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- [Milvus 2.5+](https://milvus.io/docs/install_standalone-docker.md) on `localhost:19530` (required for BM25 hybrid search)
- PostgreSQL (database created automatically on first start)
- An OpenAI-compatible LLM endpoint (Ollama / vLLM / etc.)

### Configuration

Copy and edit `backend/.env`:

```bash
cp backend/.env.example backend/.env
```

Key settings (see `.env.example` for the full list):

```yaml
# Milvus
MILVUS_URI=http://localhost:19530
COLLECTION_NAME=papers

# Embedding / Reranker
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
EMBEDDING_DEVICE=auto          # auto | cuda | cpu

# Retrieval
FETCH_K=20
TOP_K=5
RRF_K=60

# LLM (Ollama example)
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen2.5:3b
LLM_API_KEY=ollama
LLM_TEMPERATURE=0.1
MAX_RETRIES=2

# VLM (optional)
VLM_ENABLED=false
VLM_BASE_URL=http://localhost:11434/v1
VLM_MODEL=qwen-vl

# PostgreSQL
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/scholar_rag

# Observability
CHAT_TRACE=true
CHAT_TRACE_DIR=./data/traces

# PDF parsing
DOCLING_LOW_MEMORY=true
SAVE_PARSE_ARTIFACT=true
```

**GPU acceleration** (optional, for faster embedding/reranking):

```bash
cd backend
pip install -r requirements.txt
pip uninstall -y torch
pip install -r requirements-gpu.txt
# Verify: python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Set `EMBEDDING_DEVICE=cuda` in `.env`. If CUDA is unavailable, the backend falls back to CPU automatically.

### Option 1: Docker (recommended)

```bash
cp backend/.env.example backend/.env   # edit with your model endpoints
docker-compose up -d
```

Open http://localhost:8000 (backend serves built frontend).

### Option 2: Makefile

Requires Milvus and Postgres running locally.

```bash
cp backend/.env.example backend/.env
make install       # pip + npm install
make start         # build frontend + start backend at :8000
```

Development with hot reload:

```bash
make dev           # backend :8000 + frontend dev server :5173
```

### Windows (local Python)

Do **not** use `python -m uvicorn` on Windows — ProactorEventLoop breaks async PostgreSQL (`psycopg`). Use:

```powershell
cd backend
.\.venv\Scripts\python.exe run.py
```

Or double-click `backend/start-backend.bat`.

First startup downloads embedding/reranker models from Hugging Face (may take several minutes). Set `HF_LOCAL_FILES_ONLY=true` after models are cached.

### Use

1. Open http://localhost:8000
2. Upload PDF papers via the upload panel (they bind to the current session)
3. Ask questions — get cited answers streamed in real time

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | SSE streaming chat (`{query, session_id?}`) |
| `GET` | `/api/sessions` | List sessions |
| `GET` | `/api/sessions/:id/history` | Conversation history |
| `DELETE` | `/api/sessions/:id` | Delete session |
| `POST` | `/api/files/upload` | Upload PDFs (multipart) |
| `GET` | `/api/files` | List uploaded files |
| `DELETE` | `/api/files/:id` | Delete file + vectors |
| `DELETE` | `/api/collection` | Clear vector database |
| `GET` | `/api/health` | Health check |

**SSE event sequence:** `session_id` → `status` → `sub_queries` → `answer` (token stream) → `citations` → `done`

---

## Evaluation

ScholarRAG ships a **dual-layer evaluation** on a shared 20-case dataset (`backend/eval/fixtures/scope_eval_dataset.json`): 10 synthetic academic PDFs, 20 Chinese abstract-focused questions with `must_contain` gold tokens and `relevant_ids` for retrieval.

Detailed docs: [`SCOPE_EVAL_README.md`](backend/eval/fixtures/SCOPE_EVAL_README.md) · [`RETRIEVAL_EVAL.md`](backend/eval/RETRIEVAL_EVAL.md)

### Prerequisites

1. All `eval_*` PDFs indexed in Milvus (upload via UI or eval fixture PDFs)
2. For scope eval: Ollama (or configured LLM) running
3. For retrieval eval: Milvus only — no LLM needed

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\run_scope_eval.py --limit 3   # smoke test
```

### Layer 1: Retrieval evaluation

Directly calls `Retriever.retrieve` — no agent, no LLM.

```powershell
# Preview hits for labeling
.\.venv\Scripts\python.exe scripts\retrieval_eval_preview.py --id eval_01_abstract_efficiency

# Auto-label relevant_ids (optional)
.\.venv\Scripts\python.exe scripts\auto_label_relevant_ids.py

# Full run (default: 10-paper session scope, same as UI)
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py

# Single-paper retrieval scope (ablation)
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --single-paper

# Hybrid vs dense vs BM25 channel comparison
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --preset channel
```

Output: `eval/results/retrieval_eval_latest.json`

| Metric | Hybrid (default) | Notes |
|---|---|---|
| **Recall@5** | **81.7%** | dense + BM25 → RRF + rerank + parent expand |
| **MRR** | 0.77 | |
| **contamination@k** | 0.39 | fraction of top-k from other papers in multi-PDF session |

Channel ablation (same pipeline, rerank on): hybrid 81.7% > dense 75.0% > BM25 38.3%.

### Layer 2: Scope evaluation (end-to-end)

Runs the full agent graph — same code path as production chat.

```powershell
# Default: 10-paper session (matches UI uploading all eval PDFs)
.\.venv\Scripts\python.exe scripts\run_scope_eval.py

# Single-paper session (only the case's paper_id)
.\.venv\Scripts\python.exe scripts\run_scope_eval.py --single-paper

# Compare both modes back-to-back
.\.venv\Scripts\python.exe scripts\run_scope_eval.py --compare
```

Output: `eval/results/scope_eval_latest.json` or `scope_eval_compare_latest.json`

| Mode | passed | scope_ok | Description |
|---|---|---|---|
| **Single PDF** | 17/20 (85%) | 18/20 (90%) | Session scoped to one paper |
| **10 PDF session** | 12/20 (60%) | 16/20 (80%) | Same as UI multi-upload |

**Metrics:**

- **passed** — answer contains all `must_contain` tokens
- **scope_ok** — citation indices in the final answer reference only the expected `paper_id`

Multi-PDF sessions are harder on factual questions (platform names, metrics, corpus sizes) because retrieval pool spans 10 papers. Single-PDF mode improves accuracy but does not reflect the typical multi-upload UI scenario.

> Results vary with LLM temperature and model choice. Numbers above are from `scope_eval_compare_latest.json` (2026-05-26).

---

## Observability

Enable chat tracing in `.env`:

```yaml
CHAT_TRACE=true
CHAT_TRACE_SAVE=true
CHAT_TRACE_DIR=./data/traces
```

Each `/api/chat` request writes a JSON trace under `data/traces/<session_id>/<trace_id>.json` with staged events:

| Stage | Content |
|---|---|
| `classify` | query type, section_type filter |
| `retrieve` | top-k hits with paper_id, section_type, preview |
| `reflect` | sufficiency judgment, retry decisions |
| `prepare_synthesis` | pooled citations, focus_paper inference |
| `synthesize` | final answer |

Eval runs write to `data/traces/eval_<case_id>/` when `CHAT_TRACE=true`.

Console logs also emit `CHAT_TRACE` lines for quick grep during development.

---

## Tech Stack

<details>
<summary><strong>LLM Orchestration (LangGraph)</strong></summary>

Built with LangGraph (`backend/agent/graph.py`):

**Main graph:** `summarize` → `classify` → `analyze` → parallel `sub_agent` → `prepare_synthesis` → `synthesize`

- **summarize** — compresses conversation history beyond 6 turns via LLM summary + `RemoveMessage`
- **classify** — LLM structured output (`QueryClassification`); abstract queries (`摘要` / `abstract`) force `general` with no section filter
- **analyze** — decomposes into sub-queries (`QueryAnalysis`); dispatches via LangGraph `Send`
- **prepare_synthesis** — pools sub-answer citations, infers focus paper in multi-PDF sessions, remaps citation indices
- **synthesize** — streams final answer with unified `[n]` citations; fallback when LLM returns citation-only output

**Sub-agent graph:** `retrieve` → `generate` → `reflect` → retry or done

- **reflect** — `ReflectionResult` sufficiency check; triggers VLM for figure-heavy insufficient answers
- **retrieve** — applies `section_type_filter` from classify; falls back to unfiltered search on zero hits

LLM calls via `langchain-openai` `ChatOpenAI` (any OpenAI-compatible API).

</details>

<details>
<summary><strong>Vector Database (Milvus 2.5+)</strong></summary>

Milvus standalone via Docker Compose (`milvusdb/milvus:v2.5.4`).

**Hybrid retrieval** (`backend/rag/retrieval.py`):

- Dense vector + BM25 sparse indexes via `langchain-milvus` + `BM25BuiltInFunction`
- RRF fusion (`rrf_k=60`), CrossEncoder reranking, parent chunk backtracking
- Metadata filters: `paper_id`, `node_type`, `section_type`, `section_path`

**Parent-child chunking** (`backend/rag/integration.py`):

- Parent chunks = semantic units; child chunks = 500-char slices (50 overlap)
- Tables, figures, headings kept as atomic child chunks
- Collections: `{name}_children` and `{name}_parents`

```
Query → [HyDE] → Hybrid Search → RRF → Rerank → Parent Expand → Dedup → Top-K
```

Also: LRU retrieval cache, incremental updates by `paper_id`.

</details>

<details>
<summary><strong>PDF Parsing (Docling + PyMuPDF)</strong></summary>

- **Docling** — structure-aware parsing (sections, tables, figures, formulas); OCR fallback when text density is low
- **node_generator.py** — factory for 6 node types; tables linearized as `Row N: key=val`
- **PyMuPDF** — figure cropping at 2× DPI, saved to `data/figures/{paper_id}/`
- **Parse artifacts** — optional full parse JSON in `data/parsed/` when `SAVE_PARSE_ARTIFACT=true`

</details>

<details>
<summary><strong>Backend (FastAPI + SSE)</strong></summary>

- 4 routers: chat, sessions, files, manage
- SSE via `sse-starlette` `EventSourceResponse`
- Synthesis tokens streamed via `llm.astream()`
- State persisted to PostgreSQL checkpointer after each turn
- Frontend static files mounted from `frontend/dist` for single-port deployment

</details>

<details>
<summary><strong>Frontend (React + Vite)</strong></summary>

- React 18 functional components + Hooks
- `react-markdown` for AI response rendering
- `api.js` — native fetch + ReadableStream SSE parser with `AbortController` cancel
- TailwindCSS 3.4; production via Nginx (`frontend/nginx.conf`)

</details>

<details>
<summary><strong>State Persistence</strong></summary>

- PostgreSQL 16 for LangGraph checkpoints (`AsyncPostgresSaver`) and session metadata
- SQLite for file/session registry (`app/store.py`)
- In-memory checkpointer available for lightweight testing

</details>

---

## Security Notice

ScholarRAG is designed as a **research and learning tool** for **trusted local or internal network environments**. It does not include production-grade security hardening.

- **No authentication** — all endpoints are publicly accessible
- **Session IDs are the only access boundary** — anyone with a session ID can read or delete its history
- **Destructive endpoints unprotected** — `DELETE /api/collection` wipes the vector database without confirmation
- **Plaintext credentials** in `.env` — never commit `.env` to version control
- **Default passwords** (`postgres:postgres`) must be changed for non-local deployment
- **Plain HTTP** by default — use a reverse proxy with TLS for external exposure
- **CORS fully open** — restrict origins in production

> [!CAUTION]
> Do not expose ScholarRAG directly to the public internet without authentication, TLS, and access controls.

---

## License

This project is open source under the [MIT License](./LICENSE).

---

## Contributors

- [PangHu1020](https://github.com/PangHu1020)
- [curme-miller](https://github.com/curme-miller)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PangHu1020/scholar-rag&type=Date)](https://www.star-history.com/#PangHu1020/scholar-rag&Date)
