# nano_agent

轻量级 Agent HTTP 服务 — **继承 AgentScope ReActAgent，复用 QwenPaw MCP 客户端。**

## 设计哲学

```
nano_agent = AgentScope ReActAgent + QwenPaw StatefulClient + MaaS 原生对接
```

| 组件 | 来源 | 说明 |
|:----|:----|:----|
| ReAct 循环 | `agentscope.agent.ReActAgent` | 思考→行动→观察→循环，完整的 LLM 编排 |
| 模型调用 | `agentscope.model.OpenAIChatModel` | 对接任何 OpenAI 兼容 API（含 MaaS） |
| 工具管理 | `agentscope.tool.Toolkit` | 内置工具 / SKILL / MCP 统一注册 |
| 消息格式 | `agentscope.message.Msg` | AgentScope 标准消息协议 |
| SKILL 加载 | AgentScope `register_agent_skill` | `skills/<name>/SKILL.md` 文件夹式（QwenPaw 兼容） |
| MCP 客户端 | `qwenpaw.app.mcp.*StatefulClient` | stdio / sse / streamable_http |
| MCP 配置 | 自定 `mcp_config.yaml` | 字段对齐 QwenPaw MCPClientConfig |
| 会话记忆 | `SessionMemory` | 按 session_id 隔离，JSONL 持久化 |

## 项目结构

```
nano_agent/
├── server.py        # FastAPI 入口（~135 行）
├── agent.py         # NanoAgent(ReActAgent) — 工具加载 + 会话管理（~190 行）
├── config.py        # Pydantic Settings，仅 .env（~60 行）
├── memory.py        # SessionMemory — JSONL 持久化（~210 行）
├── start.sh         # 一键启动
├── .env.example     # 环境变量模板
├── mcp_config.yaml  # MCP 客户端配置（自定 YAML）
├── requirements.txt # 依赖
├── README.md        # 本文件
├── skills/          # SKILL 目录
└── memory/          # 会话数据（自动创建）
```

## 快速开始

```bash
# 1. 配置
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY 等

# 2. 启动
bash start.sh

# 3. 测试（新会话）
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，你是谁？"}'

# 4. 多轮对话（传入上一步返回的 session_id）
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我上一轮问了什么？", "session_id": "a1b2c3d4e5f6"}'

# 5. 健康检查
curl http://localhost:8080/health

# 6. 清空会话
curl -X POST http://localhost:8080/reset \
  -H "Content-Type: application/json" \
  -d '{"session_id": "a1b2c3d4e5f6"}'
```

## MCP 配置

编辑 `mcp_config.yaml`（自定 YAML 格式，字段对齐 QwenPaw MCPClientConfig）：

```yaml
clients:
  my-search:
    name: search_mcp
    transport: stdio
    command: npx
    args: ["-y", "tavily-mcp@latest"]
    env:
      TAVILY_API_KEY: "your-key-here"
```

## SKILL 配置

创建 `skills/<name>/SKILL.md`：

```markdown
---
name: my-skill
description: "技能描述"
---

# 技能正文

告诉 Agent 如何使用这个技能。
```

## API

| 端点 | 方法 | 说明 |
|:----|:----|:----|
| `/health` | GET | 工具注册摘要 |
| `/chat` | POST | 对话（支持 `session_id` 多轮） |
| `/reset` | POST | 清空指定会话记忆 |
