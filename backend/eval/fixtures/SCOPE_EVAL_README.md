# Session Scope 评测集

## 文件

| 路径 | 说明 |
|------|------|
| `pdfs_regenerated/eval_01_*.pdf` … | 测试 PDF（已生成，无需再跑脚本） |
| `scope_eval_dataset.json` | 20 道题 + `must_contain` + `relevant_ids`（检索金标准）+ 参考答案 |

## 准备

1. Milvus 中已有 `eval_*` 文档（在 UI **新建聊天** → 上传对应 PDF，或重复上传时选「已在库中」绑定会话）。
2. Ollama 已启动。

单篇问答：**一个会话只绑一篇 PDF**（与评测「每题一篇」一致）。

## 跑评测

```powershell
cd "d:\agent project two\agentic rag\agentic-rag-project\backend"

.\.venv\Scripts\python.exe scripts\run_scope_eval.py --limit 3
.\.venv\Scripts\python.exe scripts\run_scope_eval.py
```

结果：`eval/results/scope_eval_latest.json`

- **passed**：答案含 `must_contain` 各项  
- **scope_ok**：引用仅来自本题 `paper_id`  

链路日志：`data/traces/eval_<case_id>/eval_<case_id>.json`（需 `CHAT_TRACE=true`）

## 检索层评测（同一数据集）

见 [RETRIEVAL_EVAL.md](../RETRIEVAL_EVAL.md)。

```powershell
.\.venv\Scripts\python.exe scripts\retrieval_eval_preview.py --id eval_01_abstract_efficiency
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py
```

默认在 **10 篇 eval PDF 同会话** 范围内检索（与 UI 一次上传全部 PDF 一致）；`relevant_ids` 仍只标本题 `paper_id` 的 chunk。  
单篇对照加 `--single-paper`。

结果：`eval/results/retrieval_eval_latest.json`（Recall@k、MRR、`contamination@k`、消融对比）

混合 vs 单路：`python scripts/run_retrieval_eval.py --preset channel`
