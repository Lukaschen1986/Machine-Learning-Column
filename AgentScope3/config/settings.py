"""AgentScope3 配置 — 环境变量驱动，自动加载 .env"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置

    所有字段均支持环境变量覆盖（通过 alias 指定环境变量名），
    同时也会从项目根目录的 .env 文件自动加载。

    优先级：环境变量 > .env 文件 > 代码默认值
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 服务 ──────────────────────────────────────────────────
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
    )
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8080, alias="PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # ── 大模型（OpenAI 兼容） ──────────────────────────────────
    model_name: str = Field(default="deepseek-chat", alias="LLM_MODEL_NAME")
    api_base: str = Field(
        default="https://api.deepseek.com/v1",
        alias="LLM_API_BASE",
    )
    api_key: str = Field(default="", alias="LLM_API_KEY")
    temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")

    # ── Agent 行为 ────────────────────────────────────────────
    max_iters: int = Field(default=10, alias="AGENT_MAX_ITERS")
    memory_rounds: int = Field(default=20, alias="AGENT_MEMORY_ROUNDS")

    def model_post_init(self, _context: Any) -> None:
        """初始化后回退：如果 LLM_API_KEY 未设，尝试 DEEPSEEK_API_KEY"""
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_API_KEY", "")

    # ── YAML 配置文件加载 ─────────────────────────────────────

    def load_mcp_configs(self) -> list[dict]:
        """加载 config/mcp_servers.yaml，返回 MCP 服务器列表"""
        return self._load_list("config/mcp_servers.yaml") or []

    def load_prompt_config(self) -> dict:
        """加载 config/prompts.yaml，返回提示词配置"""
        data = self._load_dict("config/prompts.yaml")
        return data or {"default": _DEFAULT_PROMPT}

    def scan_skills(self) -> list[Path]:
        """扫描 skills/ 目录，返回所有含 SKILL.md 的子目录"""
        skills_dir = self.project_root / "skills"
        if not skills_dir.exists():
            return []
        return sorted(
            d for d in skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        )

    def _load_dict(self, rel: str) -> dict:
        p = self.project_root / rel
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        return {}

    def _load_list(self, rel: str) -> list | None:
        p = self.project_root / rel
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "servers" in data:
                    return data["servers"]
                return data if isinstance(data, list) else None
        return None


_DEFAULT_PROMPT = """你是 AgentScope3，一个轻量级的业务智能助手。

核心能力：
1. 工具调用 — 可以使用 MCP 工具完成外部查询和操作
2. 知识问答 — 可以使用已加载的业务技能文档回答相关问题

回答原则：
- 准确、简洁、专业
- 对于不确定的信息，如实告知
- 需要用户确认的操作，先问清楚再执行"""

settings = Settings()
