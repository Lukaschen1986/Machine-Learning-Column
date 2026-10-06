"""nano_agent — NanoAgent

继承 AgentScope 的 ReActAgent，用 QwenPaw StatefulClient 连接 MCP，
用 AgentScope Toolkit 加载 SKILL.md 技能。
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime

import mcp
import yaml
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from qwenpaw.app.mcp import HttpStatefulClient, StdIOStatefulClient

from config import get_settings
from memory import SessionMemory

logger = logging.getLogger("nano_agent.agent")


# ── 内置工具（模块级函数，Toolkit 可直接注册） ──


async def get_current_time() -> str:
    """获取当前时间，格式为 YYYY-MM-DD HH:MM:SS"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── 环境变量解析 ──

_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: str) -> str:
    """解析字符串中的 \${VAR} 环境变量引用。

    示例：
        "https://host?key=\${DIDI_MCP_KEY}" → "https://host?key=abc123"
        "https://host/path" → "https://host/path"（无变化）
    """

    def _repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in os.environ:
            logger.warning("环境变量未设置: %s", key)
        return os.environ.get(key, "")

    return _ENV_RE.sub(_repl, value)


# ── MCP 客户端工厂 ──


def _build_mcp_clients(mcp_config_path: str) -> list:
    """从 mcp_config.yaml 构建 MCP 客户端列表。

    自定 YAML 格式，字段对齐 QwenPaw MCPClientConfig。
    URL 中支持 \${ENV_VAR} 环境变量引用。

        clients:
          my-mcp:
            name: my-mcp
            transport: stdio|sse|streamable_http
            command: npx
            args: ["-y", "..."]
            env: {API_KEY: xxxx}
            url: "http://...?key=\${API_KEY}"
            headers: {Authorization: xxx}
    """
    if not mcp_config_path or not os.path.isfile(mcp_config_path):
        logger.info("未配置 MCP（%s），跳过", mcp_config_path)
        return []

    with open(mcp_config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    clients = []
    for _name, cfg in (raw or {}).get("clients", {}).items():
        transport = cfg.get("transport", "stdio")
        try:
            if transport == "stdio":
                client = StdIOStatefulClient(
                    name=cfg["name"],
                    command=_resolve_env_vars(cfg.get("command", "")),
                    args=list(cfg.get("args", [])),
                    env=dict(cfg.get("env", {})),
                    cwd=cfg.get("cwd"),
                )
            else:
                client = HttpStatefulClient(
                    name=cfg["name"],
                    transport=transport,
                    url=_resolve_env_vars(cfg.get("url", "")),
                    headers=dict(cfg.get("headers", {})),
                )
            clients.append(client)
            logger.info("MCP '%s' 已创建（transport=%s）", cfg["name"], transport)
        except Exception as e:
            logger.warning("MCP '%s' 创建失败: %s", cfg.get("name", _name), e)

    return clients


def _register_skills(toolkit: Toolkit, skills_dir: str) -> None:
    """扫描 skills_dir 下的文件夹，用 Toolkit.register_agent_skill 注册。

    兼容 QwenPaw 的 SKILL.md 格式：
        skills/<name>/
        └── SKILL.md  # 含 YAML frontmatter（name, description）
    """
    if not os.path.isdir(skills_dir):
        logger.info("SKILL 目录 '%s' 不存在，跳过", skills_dir)
        return

    for entry in sorted(os.scandir(skills_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        try:
            toolkit.register_agent_skill(entry.path)
            logger.info("SKILL 注册: %s", entry.name)
        except (ValueError, FileNotFoundError) as e:
            logger.warning("SKILL '%s' 加载失败: %s", entry.name, e)


# ── 内置工具列表 ──

_BUILTIN_TOOLS = [
    (get_current_time, "get_current_time", "获取当前时间，格式为 YYYY-MM-DD HH:MM:SS"),
]

# ── NanoAgent ──


class NanoAgent(ReActAgent):
    """轻量级 Agent — 继承 AgentScope ReActAgent。

    关键设计：
    - OpenAIChatModel 对接 MaaS（兼容 OpenAI API 的推理服务）
    - Toolkit 管理全部工具（内置/MCP/SKILL 一视同仁）
    - SKILL 加载：skills/<name>/SKILL.md（QwenPaw 兼容格式）
    - MCP 客户端：复用 QwenPaw StatefulClient，配置用自定 YAML
    - 记忆：SessionMemory 按 session_id 隔离持久化到 JSONL

    用法：
        agent = NanoAgent(settings)
        reply = await agent.reply("你好", session_id="abc123")
    """

    def __init__(
        self,
        settings=None,
        mcp_clients: list | None = None,
    ) -> None:
        if settings is None:
            settings = get_settings()
        self._settings = settings

        # 1. 模型 + 格式化器
        model = OpenAIChatModel(
            model_name=settings.llm_model_name,
            api_key=settings.llm_api_key,
            stream=False,
            client_kwargs={"base_url": settings.llm_api_base},
        )
        formatter = OpenAIChatFormatter()

        # 2. Toolkit + 内置工具
        toolkit = Toolkit()
        for func, name, desc in _BUILTIN_TOOLS:
            toolkit.register_tool_function(
                func,
                func_name=name,
                func_description=desc,
                namesake_strategy="override",
            )

        # 3. 注册 SKILL
        _register_skills(toolkit, settings.agent_skills_dir)

        # 4. MCP 客户端（外部传入或从配置文件加载）
        self._mcp_clients = mcp_clients or _build_mcp_clients(
            settings.mcp_config_path,
        )
        self._mcp_registered = False

        # 5. 初始化 ReActAgent（memory 稍后按 session 动态切换）
        super().__init__(
            name=settings.agent_name,
            sys_prompt=settings.get_sys_prompt(),
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            max_iters=settings.agent_max_iters,
        )

        # 6. SessionMemory 缓存
        self._sessions: dict[str, SessionMemory] = {}

    async def _ensure_mcp_registered(self) -> None:
        """确保 MCP 工具已注册（延迟注册，首次 reply 时触发）。"""
        if self._mcp_registered or not self._mcp_clients:
            return

        for client in self._mcp_clients:
            try:
                await self.toolkit.register_mcp_client(
                    client,
                    namesake_strategy="skip",
                )
                name = getattr(client, "name", repr(client))
                logger.info("MCP 工具注册完成: %s", name)
            except (mcp.types.ErrorData, Exception) as e:
                logger.warning(
                    "MCP '%s' 注册失败: %s",
                    getattr(client, "name", repr(client)),
                    e,
                )
        self._mcp_registered = True

    def _get_session_memory(self, session_id: str) -> SessionMemory:
        """获取或创建指定 session 的 SessionMemory。"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(
                session_id=session_id,
                memory_dir=getattr(
                    self._settings, "memory_dir", "memory",
                ),
            )
        return self._sessions[session_id]

    async def reset_session(self, session_id: str) -> int:
        """清空指定 session 的记忆，返回清除的消息数。

        若 session 不存在，返回 0（幂等）。
        """
        mem = self._sessions.pop(session_id, None)
        if mem is None:
            return 0
        count = await mem.size()
        await mem.clear()
        return count

    async def reply(
        self,
        msg: Msg | str,
        session_id: str | None = None,
    ) -> Msg:
        """统一的 reply 接口。

        Args:
            msg: 用户输入（Msg 或字符串）
            session_id: 会话标识符。传入时自动切换到对应 SessionMemory；
                        留空使用当前内存（无持久化）。

        Returns:
            AgentScope Msg
        """
        # 延迟注册 MCP（首次 reply 时触发，避免 __init__ 中的竞态）
        await self._ensure_mcp_registered()

        # 按 session 切换 memory
        if session_id:
            self.memory = self._get_session_memory(session_id)

        # 字符串快捷方式 → Msg
        if isinstance(msg, str):
            msg = Msg(self.name, msg, role="user")

        return await super().reply(msg)
