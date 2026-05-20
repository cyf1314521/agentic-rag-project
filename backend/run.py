"""
Windows 安全启动入口。

Windows + Python 3.13 下 uvicorn.run() 仍会使用 ProactorEventLoop，
导致 psycopg 异步无法连接。本脚本用 SelectorEventLoop 显式启动服务。

用法（在 backend 目录）:
  .\\.venv\\Scripts\\python.exe run.py
"""

import os
import sys
import asyncio
import selectors

# 本地 Ollama 勿走系统 HTTP 代理，否则易出现 502 Bad Gateway
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
# 嵌入/重排模型已下载过时，避免启动卡 huggingface.co（WinError 10060）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import uvicorn

from config import Config


async def _serve():
    config = uvicorn.Config(
        "app.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
    )
    server = uvicorn.Server(config)
    await server.serve()


def main():
    if sys.platform == "win32":
        asyncio.run(
            _serve(),
            loop_factory=asyncio.SelectorEventLoop,
        )
    else:
        asyncio.run(_serve())


if __name__ == "__main__":
    main()
