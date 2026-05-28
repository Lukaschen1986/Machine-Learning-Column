"""nano_agent — SessionMemory

按 session_id 隔离的 JSONL 持久化记忆。
继承 AgentScope MemoryBase，对接 ReActAgent 的 memory 参数。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from agentscope.memory import MemoryBase
from agentscope.message import Msg

logger = logging.getLogger("nano_agent.memory")

# ── session_id 安全校验 ──

_VALID_SESSION_ID = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")


def validate_session_id(sid: str) -> bool:
    """防路径穿越 — session_id 只允许字母数字和连字符"""
    return bool(_VALID_SESSION_ID.match(sid))


# ── SessionMemory ──


class SessionMemory(MemoryBase):
    """按 session_id 隔离的 JSONL 持久化记忆。

    每个 session 对应 memory/ 下一个独立 JSONL 文件。
    格式：每行一条 {"msg": {...}, "marks": [...]}

    用法：
        mem = SessionMemory("abc123")
        await mem.add(Msg("user", "hello", "user"))
        history = await mem.get_memory()
    """

    def __init__(
        self,
        session_id: str,
        memory_dir: str = "memory",
    ) -> None:
        """初始化 SessionMemory。

        Args:
            session_id: 会话标识符（需通过 validate_session_id 校验）
            memory_dir: JSONL 文件存放目录

        Raises:
            ValueError: session_id 格式不合法
        """
        super().__init__()

        if not validate_session_id(session_id):
            raise ValueError(
                f"session_id 格式不合法: '{session_id}'。"
                f"只允许字母、数字、连字符和下划线，6-64 字符。",
            )

        self.session_id = session_id
        self.memory_dir = memory_dir
        self.file_path = os.path.join(memory_dir, f"{session_id}.jsonl")

        # 与 InMemoryMemory 保持相同的数据结构
        self.content: list[tuple[Msg, list[str]]] = []

        self._load()
        self.register_state("content")

    # ── 持久化 ──

    def _load(self) -> None:
        """从 JSONL 加载历史消息。

        格式：
            {"msg": {"name": "...", "content": "...", "role": "..."}, "marks": ["..."]}
        """
        if not os.path.isfile(self.file_path):
            logger.debug("会话 '%s' 无历史文件", self.session_id)
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    msg = Msg.from_dict(record["msg"])
                    marks = record.get("marks", [])
                    self.content.append((msg, marks))
            logger.info(
                "会话 '%s' 加载 %d 条历史",
                self.session_id,
                len(self.content),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "会话 '%s' JSONL 解析失败: %s，从空记忆启动",
                self.session_id,
                e,
            )
            self.content = []

    def _save(self) -> None:
        """将当前 content 写入 JSONL。

        TODO: 消息量大时改为追加写 + 定期压缩，避免全量覆盖。
        """
        os.makedirs(self.memory_dir, exist_ok=True)

        with open(self.file_path, "w", encoding="utf-8") as f:
            for msg, marks in self.content:
                record = {"msg": msg.to_dict(), "marks": marks}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── MemoryBase 抽象方法 ──

    async def add(
        self,
        memories: Msg | list[Msg] | None,
        marks: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """添加消息到记忆并持久化。

        Args:
            memories: 单条或多条消息
            marks: 标记（用于分类检索）
        """
        if memories is None:
            return

        if isinstance(memories, Msg):
            memories = [memories]

        if isinstance(marks, str):
            marks = [marks]
        elif marks is None:
            marks = []

        for msg in memories:
            # 跳过空消息
            if not msg.content:
                continue
            self.content.append((msg, marks))

        self._save()

    async def delete(
        self,
        msg_ids: list[str],
        **kwargs: Any,
    ) -> int:
        """按消息 ID 删除。

        Returns:
            删除的条数
        """
        removed = 0
        new_content = []
        for msg, marks in self.content:
            if msg.id in msg_ids:
                removed += 1
            else:
                new_content.append((msg, marks))
        self.content = new_content
        self._save()
        return removed

    async def delete_by_mark(
        self,
        mark: str | list[str],
        **kwargs: Any,
    ) -> int:
        """按标记删除消息。

        Returns:
            删除的条数
        """
        if isinstance(mark, str):
            mark = [mark]
        mark_set = set(mark)

        removed = 0
        new_content = []
        for msg, marks in self.content:
            if mark_set & set(marks):
                removed += 1
            else:
                new_content.append((msg, marks))
        self.content = new_content
        self._save()
        return removed

    async def size(self) -> int:
        """记忆中的消息数。"""
        return len(self.content)

    async def clear(self) -> None:
        """清空记忆。"""
        self.content = []
        self._save()

    async def get_memory(
        self,
        mark: str | None = None,
        exclude_mark: str | None = None,
        prepend_summary: bool = True,
        **kwargs: Any,
    ) -> list[Msg]:
        """检索记忆。

        Args:
            mark: 只返回带有此标记的消息（None = 全部）
            exclude_mark: 排除带有此标记的消息
            prepend_summary: 是否在开头插入压缩摘要

        Returns:
            消息列表
        """
        # 过滤
        filtered = self.content
        if mark is not None:
            filtered = [(m, ms) for m, ms in filtered if mark in ms]
        if exclude_mark is not None:
            filtered = [(m, ms) for m, ms in filtered if exclude_mark not in ms]

        result = [msg for msg, _ in filtered]

        # 可选：prepend 压缩摘要（暂不使用）
        if prepend_summary and self._compressed_summary:
            result.insert(0, Msg("user", self._compressed_summary, "user"))

        return result

    async def update_messages_mark(
        self,
        msg_ids: list[str],
        mark: str,
        mode: str = "add",
        **kwargs: Any,
    ) -> None:
        """更新消息标记。

        Args:
            msg_ids: 消息 ID 列表
            mark: 标记
            mode: "add"（追加）或 "remove"（移除）
        """
        id_set = set(msg_ids)
        for i, (msg, marks) in enumerate(self.content):
            if msg.id in id_set:
                if mode == "add" and mark not in marks:
                    marks.append(mark)
                elif mode == "remove" and mark in marks:
                    marks.remove(mark)

    # ── StateModule 序列化 ──

    def state_dict(self) -> dict:
        """导出状态（供 AgentScope 框架使用）。"""
        return {
            "content": [
                (msg.to_dict(), list(marks))
                for msg, marks in self.content
            ],
            "_compressed_summary": self._compressed_summary,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """恢复状态。"""
        self.content = [
            (Msg.from_dict(msg_dict), list(marks))
            for msg_dict, marks in state_dict.get("content", [])
        ]
        self._compressed_summary = state_dict.get("_compressed_summary", "")
