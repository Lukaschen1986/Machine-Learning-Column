"""nano_agent — FastAPI 服务入口

最小化的 HTTP 包装：启动 NanoAgent（继承 ReActAgent），对外提供 /chat 接口。
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import NanoAgent
from config import get_settings
from memory import validate_session_id

logger = logging.getLogger("nano_agent.server")

# ── 全局变量（lifespan 中初始化） ──

_agent = None
_settings = None


# ── 请求/响应模型 ──


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8192)
    session_id: str = Field("", description="留空自动生成")


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=6, max_length=64)


class ResetResponse(BaseModel):
    session_id: str
    cleared: int


class HealthResponse(BaseModel):
    status: str
    tools: int = 0
    skills: int = 0


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _settings
    _settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, _settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    logger.info("nano_agent 启动 | 模型: %s", _settings.llm_model_name)
    logger.info("MaaS 端点: %s", _settings.llm_api_base)

    # 实例化 NanoAgent
    _agent = NanoAgent(settings=_settings)

    logger.info("nano_agent 就绪 | 工具数: %d", len(_agent.toolkit.tools))

    # 启动后在 lifespan 内动态配置 CORS（而非模块 import 时读取 None）
    _setup_cors(app)

    yield

    logger.info("nano_agent 关闭")


def _setup_cors(app: FastAPI) -> None:
    """在 lifespan 中动态配置 CORS（此时 _settings 已初始化）。"""
    origins = (
        _settings.cors_origins.split(",")
        if _settings and _settings.cors_origins != "*"
        else ["*"]
    )
    # 移除旧的 CORS middleware（如有）并重新添加
    app.user_middleware = [
        mw for mw in app.user_middleware
        if mw.cls != CORSMiddleware
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS origins: %s", origins)


# ── FastAPI 应用 ──

app = FastAPI(
    title="nano_agent",
    description="轻量级 Agent HTTP 服务（继承 AgentScope ReActAgent）",
    version="0.2.0",
    lifespan=lifespan,
)


# ── 路由 ──


@app.get("/health", response_model=HealthResponse)
async def health():
    if _agent is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return HealthResponse(
        status="ok",
        tools=len(_agent.toolkit.tools),
        skills=len(_agent.toolkit.skills),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    # 生成或校验 session_id
    session_id = req.session_id if req.session_id else uuid.uuid4().hex[:12]
    if not validate_session_id(session_id):
        raise HTTPException(
            status_code=400,
            detail=f"session_id 格式不合法: '{session_id}'",
        )

    try:
        reply = await _agent.reply(req.message, session_id=session_id)
        text = reply.content if hasattr(reply, "content") else str(reply)
        return ChatResponse(session_id=session_id, reply=text)
    except Exception as e:
        logger.exception("对话处理异常")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest):
    """清空指定会话记忆。"""
    if _agent is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    if not validate_session_id(req.session_id):
        raise HTTPException(
            status_code=400,
            detail=f"session_id 格式不合法: '{req.session_id}'",
        )

    count = await _agent.reset_session(req.session_id)
    return ResetResponse(session_id=req.session_id, cleared=count)


# ── 直接运行 ──

if __name__ == "__main__":
    cfg = _settings or get_settings()
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
    )
