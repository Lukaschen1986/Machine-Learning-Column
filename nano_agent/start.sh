#!/bin/bash
# ═══════════════════════════════════════════════════════╗
#  nano_agent — 一键启动脚本                             ║
#  用法: bash start.sh [--dev]                          ║
#  说明: 加载 .env → 检查依赖 → 启动 FastAPI 服务       ║
# ═══════════════════════════════════════════════════════╝

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 加载环境变量 ──
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "[✓] 已加载 .env"
else
    echo "[!] 未找到 .env，从 .env.example 复制一份"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "[!] 请编辑 .env 填入 LLM_API_KEY 等配置后重新执行"
        exit 1
    fi
fi

# ── 检查必要环境变量 ──
: "${LLM_API_BASE:?  请设置 LLM_API_BASE}"
: "${LLM_API_KEY:?   请设置 LLM_API_KEY}"

# ── 安装依赖（首次） ──
if [ ! -d "venv" ]; then
    echo "[...] 创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
    echo "[✓] 依赖安装完成"
else
    source venv/bin/activate
fi

# ── 启动服务 ──
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "[→] 启动 nano_agent  http://${HOST}:${PORT}"
exec uvicorn server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
