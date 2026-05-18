"""
FastAPI 应用入口。

职责：
1. 创建 FastAPI 实例并注册 lifespan（启动时加载模型单例）
2. 挂载 CORS、业务路由（chat / sessions / files / manage）
3. 若存在 frontend/dist，则挂载静态资源实现前后端同端口部署
"""

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import Config
from app.dependencies import lifespan
from app.routers import chat, sessions, files, manage

# 全局日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)

# lifespan 在启动/关闭时初始化 LLM、Retriever、Postgres 等单例
app = FastAPI(title="Scholar RAG", lifespan=lifespan)

# 开发模式：允许任意来源跨域（生产环境应收紧 allow_origins）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册各业务路由模块
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(files.router)
app.include_router(manage.router)

# 生产构建后可将 React 静态文件挂载到根路径
dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")


if __name__ == "__main__":
    # 直接 python -m app.main 时以热重载方式启动
    uvicorn.run("app.main:app", host=Config.HOST, port=Config.PORT, reload=True)
