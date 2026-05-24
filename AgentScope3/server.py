"""
AgentScope3 — FastAPI 入口

基于 QwenPaw + AgentScope 构建的轻量级 Agent 推理服务。

启动方式：
    python server.py
    bash start.sh

API：
    GET  /health    — 健康检查
    POST /chat      — 对话（多轮会话）
    POST /reset     — 清除会话记忆
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.service import AgentService
from config.settings import settings

logger = logging.getLogger("agentscope3.server")

# 全局服务实例，由 lifespan 管理生命周期
_service: AgentService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动初始化 → 运行 → 优雅关闭"""
    global _service
    logger.info("🚀 AgentScope3 启动 — model=%s", settings.model_name)
    _service = AgentService()
    await _service.initialize()
    yield
    if _service:
        await _service.shutdown()
    logger.info("🛑 关闭")


app = FastAPI(
    title="AgentScope3",
    description="轻量级 Agent 推理服务（MCP + 技能 + 多轮对话）",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — 通过环境变量 CORS_ORIGINS 控制（默认全开）
if settings.cors_origins:
    origins = (
        ["*"]
        if settings.cors_origins == "*"
        else [o.strip() for o in settings.cors_origins.split(",")]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class ChatRequest(BaseModel):
    """对话请求体"""
    message: str = Field(..., min_length=1, description="用户输入")
    session_id: str = Field(default="", description="会话标识，留空自动创建")


class ChatResponse(BaseModel):
    """对话响应体"""
    session_id: str = Field(..., description="会话标识")
    reply: str = Field(..., description="Agent 回复")
    success: bool = True


class ResetRequest(BaseModel):
    """清除记忆请求体"""
    session_id: str = Field(..., min_length=1, description="要清除的会话 ID")


@app.get("/health")
async def health() -> dict:
    """健康检查 — 返回服务状态、MCP 连接、技能加载等"""
    srv = _service
    if not srv:
        return {"status": "init"}
    return {"status": "ok" if srv.status["ready"] else "loading",
            **srv.status}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    """对话接口

    首次请求可不传 session_id，服务端自动创建。
    后续请求传入相同 session_id 延续多轮对话。
    """
    srv = _service
    if not srv:
        raise HTTPException(503, "服务尚未就绪")
    sid = body.session_id.strip() or uuid.uuid4().hex[:12]
    reply = await srv.chat(sid, body.message)
    return ChatResponse(session_id=sid, reply=reply)


@app.post("/reset")
async def reset(body: ResetRequest) -> dict:
    """清除指定会话的记忆（对话历史）"""
    srv = _service
    if not srv:
        raise HTTPException(503, "服务尚未就绪")
    ok = srv.clear_memory(body.session_id)
    return {"success": ok, "session_id": body.session_id}


def main():
    """直接运行入口（等效 bash start.sh）"""
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
