# AgentScope3 — 轻量级 Agent 服务

基于 **QwenPaw** + **AgentScope** 构建，承接知识问答、智能助手、业务文档咨询类任务。

## 核心能力

| 能力 | 实现 |
|:----|:------|
| **MCP 工具** | 百度地图 + 滴滴出行（`qwenpaw.app.mcp` 客户端） |
| **业务技能** | `skills/` 目录放 SKILL.md，重启即生效 |
| **多轮对话** | 文件级记忆，`memory/{session_id}.jsonl`（JSONL 格式） |
| **模型可配** | 任意 OpenAI 兼容 API（环境变量切换） |
| **生产部署** | ModelArts 一键启动（`bash start.sh`） |

## 项目结构

```
AgentScope3/
├── server.py            ← FastAPI 入口（3 个 API）
├── agent/
│   └── service.py       ← 核心服务（MCP + 技能 + 对话）
├── config/
│   ├── settings.py      ← Pydantic 配置（环境变量覆盖）
│   ├── mcp_servers.yaml ← MCP 服务器注册
│   └── prompts.yaml     ← 系统提示词
├── skills/
│   └── didi-ride-skill/ ← 示例技能
├── memory/              ← 对话记忆（自动创建）
├── start.sh             ← 一键启动
├── .env.example         ← 环境变量模板
├── requirements.txt     ← 依赖清单
└── README.md
```

## 快速开始

```bash
# 1. 配置
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 2. 启动（自动安装依赖）
bash start.sh

# 或手动
python server.py
```

## API 文档

| 方法 | 路径 | 说明 |
|:----|:----|:------|
| `GET` | `/health` | 健康检查 + 状态 |
| `POST` | `/chat` | 对话（自动创建/延续会话） |
| `POST` | `/reset` | 清除会话记忆 |

### 对话示例

```bash
# 第1轮（自动创建 session）
curl -X POST http://localhost:8080/chat \
  -d '{"message": "从福润雅居到南京南站打车多少钱"}' | json_pp

# 第2轮（延续会话）
curl -X POST http://localhost:8080/chat \
  -d '{"message": "那坐地铁呢", "session_id": "上一轮返回的id"}' | json_pp
```

## 添加业务技能

1. 创建技能文件夹：
```bash
mkdir -p skills/my-business-doc
```

2. 编写 `SKILL.md`：
```markdown
---
name: my-business-doc
description: 业务文档知识库
---

# 我的业务文档

产品 A 的核心功能是...
```

3. 重启服务：
```bash
bash start.sh
```

## 模型配置

通过环境变量切换模型：

```bash
# 使用 DeepSeek
export LLM_MODEL_NAME=deepseek-chat
export LLM_API_BASE=https://api.deepseek.com/v1
export DEEPSEEK_API_KEY=your_key

# 使用本地模型
export LLM_MODEL_NAME=qwen3.5
export LLM_API_BASE=http://127.0.0.1:8080/v1
export LLM_API_KEY=not-needed

bash start.sh
```

## 已知限制

- **无长期记忆**：重启后对话历史重置（`memory/*.jsonl` 文件可保留，需自行挂载持久化存储）
- **单进程模型**：不适用于高并发场景
- **首次启动需下载依赖**：ModelArts 受限网络环境可能因 `pip install` 超时，建议预装依赖或使用自定义镜像
