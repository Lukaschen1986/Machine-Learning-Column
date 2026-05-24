"""
Agent 服务 — MCP 连接 + 技能注册 + 多轮对话

依赖：
- QwenPaw：MCP 客户端（HttpStatefulClient / StdIOStatefulClient）
- AgentScope：ReActAgent, Toolkit, OpenAIChatModel, Msg
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from qwenpaw.app.mcp import HttpStatefulClient, StdIOStatefulClient

from config.settings import settings

logger = logging.getLogger("agentscope3.service")

# session_id 只允许字母数字和连字符
_VALID_SID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class AgentService:
    """Agent 服务（单例）

    职责：
    - 管理 MCP 客户端连接与自动重连
    - 加载并注册 skills/ 目录下的业务技能
    - 提供多轮对话接口（JSONL 文件记忆）
    """

    def __init__(self) -> None:
        self._model: Optional[OpenAIChatModel] = None
        self._formatter: Optional[OpenAIChatFormatter] = None
        self._toolkit: Optional[Toolkit] = None
        self._sys_prompt = ""
        self._mcp_clients: list = []
        self._mcp_status: dict[str, str] = {}
        self._skills: list[str] = []
        self._ready = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._mem_dir: Optional[Path] = None

    # ═══════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════

    async def initialize(self) -> None:
        """全流程初始化：模型 → MCP → 技能 → 提示词 → 就绪"""
        logger.info("初始化...")
        self._model = self._create_model()
        self._formatter = OpenAIChatFormatter(
            promote_tool_result_images=True,
        )
        self._toolkit = Toolkit()
        self._mem_dir = settings.project_root / "memory"
        self._mem_dir.mkdir(parents=True, exist_ok=True)

        await self._connect_mcp()
        self._register_skills()
        self._sys_prompt = self._build_sys_prompt()

        # 后台 MCP 重连（每 60s 检查一次）
        self._reconnect_task = asyncio.create_task(
            self._mcp_reconnect_loop(),
        )

        self._ready = True
        logger.info(
            "就绪 — MCP: %d 已连 | 技能: %d | 工具: %d | %s",
            sum(1 for s in self._mcp_status.values() if s == "connected"),
            len(self._skills),
            len(self._toolkit.tools),
            settings.model_name,
        )

    def _create_model(self) -> OpenAIChatModel:
        """创建大模型实例（非流式，避免并发冲突）"""
        return OpenAIChatModel(
            model_name=settings.model_name,
            api_key=settings.api_key,
            stream=False,
            client_kwargs={"base_url": settings.api_base}
            if settings.api_base else None,
            generate_kwargs={
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
            },
        )

    # ═══════════════════════════════════════════════════════════
    # MCP
    # ═══════════════════════════════════════════════════════════

    async def _connect_mcp(self) -> None:
        """遍历 mcp_servers.yaml，连接所有启用且可达的 MCP 服务器"""

        configs = settings.load_mcp_configs()
        if not configs:
            return

        async def _connect_one(cfg: dict) -> None:
            name = cfg.get("name", "?")
            if not cfg.get("enabled", True):
                self._mcp_status[name] = "disabled"
                return
            try:
                client = self._build_client(cfg)
                await client.connect()
                await self._toolkit.register_mcp_client(client)
                self._mcp_clients.append(client)
                self._mcp_status[name] = "connected"
                logger.info("MCP [%s] ✅ (%s)", name, cfg.get("transport"))
            except Exception as exc:
                self._mcp_status[name] = f"failed: {exc}"
                logger.warning("MCP [%s] ❌ %s", name, exc)

        await asyncio.gather(*[_connect_one(c) for c in configs])

    @staticmethod
    def _build_client(cfg: dict) -> HttpStatefulClient | StdIOStatefulClient:
        """根据配置创建 MCP 客户端实例"""
        transport = cfg.get("transport", "streamable_http")
        if transport in ("streamable_http", "sse"):
            return HttpStatefulClient(
                name=cfg["name"],
                transport=transport,
                url=cfg["url"],
                headers=cfg.get("headers") or None,
                timeout=cfg.get("timeout", 30),
            )
        return StdIOStatefulClient(
            name=cfg["name"],
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env", {}),
        )

    async def _mcp_reconnect_loop(self) -> None:
        """后台定期检查断连的 MCP 客户端并尝试重连

        每 60s 扫描一次，仅对初始化阶段已标识为 failed 的
        连接进行重试。运行时断连的检测需更复杂的健康检查机制，
        未来可按需扩展。
        """
        while True:
            await asyncio.sleep(60)
            for name, status in list(self._mcp_status.items()):
                if status in ("connected", "disabled"):
                    continue
                logger.info("MCP [%s] 尝试重连...", name)
                try:
                    configs = settings.load_mcp_configs()
                    cfg = next(
                        (c for c in configs if c.get("name") == name),
                        None,
                    )
                    if not cfg:
                        continue
                    client = self._build_client(cfg)
                    await client.connect()
                    await self._toolkit.register_mcp_client(client)
                    self._mcp_clients.append(client)
                    self._mcp_status[name] = "connected"
                    logger.info("MCP [%s] 重连 ✅", name)
                except Exception as exc:
                    logger.warning("MCP [%s] 重连 ❌ %s", name, exc)

    # ═══════════════════════════════════════════════════════════
    # 技能
    # ═══════════════════════════════════════════════════════════

    def _register_skills(self) -> None:
        """扫描 skills/ 目录，注册所有含 SKILL.md 的文件夹"""
        for skill_dir in settings.scan_skills():
            try:
                self._toolkit.register_agent_skill(str(skill_dir))
                self._skills.append(skill_dir.name)
                logger.info("技能 [%s] ✅", skill_dir.name)
            except Exception as exc:
                logger.error("技能 [%s] ❌ %s", skill_dir.name, exc)

    # ═══════════════════════════════════════════════════════════
    # 系统提示词
    # ═══════════════════════════════════════════════════════════

    def _build_sys_prompt(self) -> str:
        """构建系统提示词：基础定义 + 技能注入"""
        prompt_cfg = settings.load_prompt_config()
        prompt = prompt_cfg.get("default", _DEFAULT_PROMPT)
        if self._toolkit:
            skill_prompt = self._toolkit.get_agent_skill_prompt()
            if skill_prompt:
                prompt += "\n\n" + skill_prompt
        return prompt

    # ═══════════════════════════════════════════════════════════
    # 对话
    # ═══════════════════════════════════════════════════════════

    async def chat(self, session_id: str, message: str) -> str:
        """多轮对话：读取记忆 → 创建 Agent → 推理 → 存入记忆

        Args:
            session_id: 会话标识（必须匹配 ^[a-zA-Z0-9_-]{{1,64}}$）
            message: 用户输入文本

        Returns:
            Agent 回复文本
        """
        if not self._ready:
            return "服务初始化中，请稍后再试"
        if not _VALID_SID.match(session_id):
            return "无效的会话 ID"

        # session 级锁 → 防止同一会话并发写入导致记忆错乱
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()

        async with self._session_locks[session_id]:
            agent = ReActAgent(
                name="AgentScope3",
                sys_prompt=self._sys_prompt,
                model=self._model,
                formatter=self._formatter,
                toolkit=self._toolkit,
                max_iters=settings.max_iters,
            )
            history = self._load_memory(session_id)
            msgs = history + [
                Msg(name="user", content=message, role="user"),
            ]
            try:
                reply = await asyncio.wait_for(
                    self._call_with_retry(agent, msgs),
                    timeout=120,
                )
                reply_text = str(reply) if reply else ""
            except asyncio.TimeoutError:
                reply_text = "处理超时，请重试"
            except Exception as exc:
                logger.exception("推理异常 [%s]", session_id)
                reply_text = f"处理出错: {exc}"

            self._save_memory(session_id, message, reply_text)
            return reply_text

    @staticmethod
    async def _call_with_retry(agent: ReActAgent,
                                msgs: list,
                                max_retries: int = 3) -> str:
        """带指数退避重试的模型调用

        Args:
            agent: ReActAgent 实例
            msgs: 消息列表
            max_retries: 最大重试次数（默认 3）

        Returns:
            模型回复消息
        """
        last_err = None
        for attempt in range(max_retries):
            try:
                return await agent(msgs)
            except Exception as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    logger.warning(
                        "重试 %d/%d（%ds 后）...",
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
        raise last_err  # type: ignore[misc]

    # ═══════════════════════════════════════════════════════════
    # 记忆（JSONL 格式）
    # ═══════════════════════════════════════════════════════════

    def _load_memory(self, session_id: str) -> list:
        """从 JSONL 文件加载历史消息

        格式：每行一个 JSON 对象，含 role / content / ts 字段
        返回最近的 N 轮对话（由 AGENT_MEMORY_ROUNDS 控制）
        """
        path = self._mem_dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        msgs: list = []
        try:
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                entry = json.loads(line)
                msgs.append(
                    Msg(
                        name=entry["role"],
                        content=entry["content"],
                        role=entry["role"],
                    ),
                )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("记忆解析失败 [%s]: %s", session_id, exc)
        # 截取最近 N 轮（每轮 1 user + 1 assistant = 2 条）
        max_msgs = settings.memory_rounds * 2
        return msgs[-max_msgs:]

    def _save_memory(self, session_id: str, user_msg: str, reply: str) -> None:
        """以 JSONL 格式追加一轮对话到记忆文件"""
        path = self._mem_dir / f"{session_id}.jsonl"
        now = datetime.now().isoformat()
        lines = [
            json.dumps(
                {"role": "user", "content": user_msg, "ts": now},
                ensure_ascii=False,
            ),
            json.dumps(
                {"role": "assistant", "content": reply, "ts": now},
                ensure_ascii=False,
            ),
        ]
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def clear_memory(self, session_id: str) -> bool:
        """清除指定会话的记忆文件

        Args:
            session_id: 要清除的会话标识

        Returns:
            True 表示存在并已删除，False 表示不存在或 ID 无效
        """
        if not _VALID_SID.match(session_id):
            return False
        path = self._mem_dir / f"{session_id}.jsonl"
        if path.exists():
            path.unlink()
            self._session_locks.pop(session_id, None)
            return True
        return False

    # ═══════════════════════════════════════════════════════════
    # 生命周期
    # ═══════════════════════════════════════════════════════════

    async def shutdown(self) -> None:
        """优雅关闭：停重连任务 → 关闭所有 MCP 连接"""
        self._ready = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        for client in self._mcp_clients:
            try:
                await client.close()
            except Exception:
                pass
        self._mcp_clients.clear()
        self._mcp_status.clear()
        logger.info("关闭完成")

    # ═══════════════════════════════════════════════════════════
    # 状态
    # ═══════════════════════════════════════════════════════════

    @property
    def status(self) -> dict:
        """服务状态摘要（用于 /health 接口）"""
        mcp_connected = {
            k: v for k, v in self._mcp_status.items() if v == "connected"
        }
        mcp_failed = {
            k: v
            for k, v in self._mcp_status.items()
            if v not in ("connected", "disabled")
        }
        return {
            "ready": self._ready,
            "model": settings.model_name,
            "mcp_connected": list(mcp_connected.keys()),
            "mcp_failed": mcp_failed,
            "skills": self._skills,
            "total_tools": len(self._toolkit.tools)
            if self._toolkit else 0,
        }


_DEFAULT_PROMPT = """你是 AgentScope3，一个轻量级的业务智能助手。

核心能力：
1. 工具调用 — 可以使用 MCP 工具完成外部查询和操作
2. 知识问答 — 可以使用已加载的业务技能文档回答相关问题

回答原则：
- 准确、简洁、专业
- 对于不确定的信息，如实告知
- 需要用户确认的操作，先问清楚再执行"""
