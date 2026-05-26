# 检索层评测（scope_eval_dataset.json）

与端到端 `run_scope_eval.py` 共用 **同一数据集**；检索评测只调用 `Retriever.retrieve`，不经过 Agent。

## 默认检索范围（与 UI 一致）

**默认**：在一个会话里上传全部 10 篇 eval PDF 的场景——每题检索时 `paper_id_filter` 为 **全部 eval 论文**，与 `chat.py` 里 `get_session_paper_ids` 返回多篇时相同。

- 标注 `relevant_ids` 时：只抄 **本题 `paper_id`** 下的 `chunk_id`（预览里带 `*** OTHER PAPER ***` 的不要标）。
- 报告里会多出 **`contamination@k`**：Top-k 里有多少条来自「别的论文」。

若要做「每题只搜一篇」的对照实验，加参数 **`--single-paper`**。

## 准备

1. Milvus 中已入库全部 `eval_*` PDF。
2. 无需 Ollama（除非 `--mode llm`）。
3. 在 `backend` 目录、已激活 `.venv` 下执行。

## 1. 标注 `relevant_ids`

**自动标注（20 题，10 篇 session 检索，按 must_contain 选本题 chunk）：**

```powershell
.\.venv\Scripts\python.exe scripts\auto_label_relevant_ids.py
```

**人工抽查 / 单题预览：**

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\retrieval_eval_preview.py --id eval_01_abstract_efficiency
.\.venv\Scripts\python.exe scripts\retrieval_eval_preview.py --limit 5
```

输出示例：`filter: session (10 papers) → ['eval_01_solar_pv', 'eval_02_ocean_ph', ...]`

编辑 `eval/fixtures/scope_eval_dataset.json`：

```json
"relevant_ids": ["37feb697-0b8e-4504-a710-6dd103b8641a"]
```

（仅本题论文的 chunk_id，即使检索池里有 10 篇。）

## 2. 跑检索评测

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --limit 3
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py
```

结果：`eval/results/retrieval_eval_latest.json`

### 混合 vs 单路检索（BM25 / 向量 / RRF 融合）

在**相同** rerank + parent 管道下，只改 Milvus 检索通道：

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --preset channel --out eval/results/retrieval_channel_eval_latest.json
```

| 配置名 | rerank | 检索通道 |
|--------|--------|----------|
| `channel_hybrid` | 是 | dense + BM25 → RRF（线上默认） |
| `channel_dense` | 是 | 仅稠密向量 |
| `channel_bm25` | 是 | 仅 BM25 sparse |
| `channel_*_raw` | 否 | 同上，只看 Milvus 融合/单路阶段 |

**解读**（10 篇同会话，`retrieval_channel_eval_latest.json`）：

| 配置 | recall@5 | mrr | 说明 |
|------|----------|-----|------|
| `channel_hybrid` | **81.7%** | 0.77 | RRF + rerank（线上默认） |
| `channel_dense` | 75.0% | 0.77 | 仅向量 |
| `channel_bm25` | 38.3% | 0.40 | 仅 BM25（中文问句受 analyzer 限制） |
| `channel_hybrid_raw` | 35.8% | 0.33 | 无 rerank 的 RRF |
| `channel_dense_raw` | 25.8% | 0.34 | 无 rerank 单向量 |
| `channel_bm25_raw` | 38.3% | 0.29 | 无 rerank 单 BM25 |

混合检索在完整管道下优于单路；中文摘要场景 BM25 单路较弱，RRF 融合 + CrossEncoder 能拉回 dense 漏掉的块。`dense`/`bm25` 单路通过 `client.search` 真·单字段实现（不再走 langchain 双路 hybrid）。

管道里 rerank 的价值见默认 `full` vs `no_rerank` 消融。

### 管道消融（默认四个）

| 配置名 | rerank | expand_parent |
|--------|--------|---------------|
| `full` | 是 | 是（线上默认） |
| `no_rerank` | 否 | 是 |
| `no_parent` | 是 | 否 |
| `minimal` | 否 | 否 |

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --configs full,no_rerank
```

### 单篇对照（可选）

```powershell
.\.venv\Scripts\python.exe scripts\retrieval_eval_preview.py --single-paper --id eval_01_abstract_efficiency
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --single-paper
```

### 无标注时用 LLM 判 hit（可选）

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --mode llm --configs full
```

## 指标说明

| 指标 | 含义 |
|------|------|
| `recall@k` | 金标准 chunk 有多少出现在 Top-k（10 篇池子里能否仍召回答案块） |
| `precision@k` | Top-k 中命中金标准的比例 |
| `mrr` | 第一个相关结果排名的倒数 |
| `map` | 平均精度均值 |
| `contamination@k` | Top-k 中 `paper_id` ≠ 本题 的比例（多 PDF 会话关键指标） |

若存在 `eval/results/scope_eval_latest.json`，`per_case` 会附带 `scope_passed` 便于对齐分析。

## 与 scope 端到端对比

| | scope eval（UI 多 PDF 会话） | retrieval eval（默认） |
|--|---------------------------|-------------------------|
| 检索范围 | 会话内全部 paper | 全部 eval paper（同上） |
| 判对 | `must_contain` | `relevant_ids` 命中 |

检索 recall=0 且 scope fail → 优先改检索；检索 recall=1 但 scope fail → 优先改生成/合成。
