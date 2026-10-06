"""nano_agent — Pydantic 配置（仅 .env，无 YAML）"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── HTTP 服务 ──
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    cors_origins: str = "*"

    # ── LLM / MaaS ──
    llm_api_base: str = "https://maas-api.example.com"
    llm_api_key: str = ""
    llm_model_name: str = "deepseek-chat"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = 4096

    # ── Agent ──
    agent_name: str = "nano"
    agent_sys_prompt_file: str = "AGENTS.md"
    agent_max_iters: int = 10
    agent_skills_dir: str = "skills"

    # ── 记忆 ──
    memory_dir: str = "memory"

    # ── MCP ──
    mcp_config_path: str = "mcp_config.yaml"

    @field_validator("mcp_config_path", mode="before")
    @classmethod
    def resolve_mcp_path(cls, v: str) -> str:
        if v and not os.path.isabs(v):
            return os.path.abspath(v)
        return v

    @field_validator("llm_api_base", mode="after")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        """确保 llm_api_base 末尾不带 /v1（由 OpenAIChatModel 内部拼接）。"""
        return v.rstrip("/").removesuffix("/v1")

    def get_sys_prompt(self) -> str:
        """读取 agent_sys_prompt_file 作为系统提示词。"""
        with open(
            self.agent_sys_prompt_file, "r", encoding="utf-8"
        ) as f:
            return f.read()


_SINGLETON: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _SINGLETON
    if _SINGLETON is None or reload:
        _SINGLETON = Settings()
    return _SINGLETON
