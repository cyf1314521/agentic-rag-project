# Session Scope 评测集

## 文件

| 路径 | 说明 |
|------|------|
| `pdfs_regenerated/eval_01_*.pdf` … | 测试 PDF（已生成，无需再跑脚本） |
| `scope_eval_dataset.json` | 20 道题 + `must_contain` + 参考答案 |

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
