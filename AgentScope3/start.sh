#!/bin/bash
# ================================================================
# AgentScope3 — 启动脚本（华为云 ModelArts 兼容）
#
# 使用方式：
#   bash start.sh                              # 默认启动
#   DEEPSEEK_API_KEY=xxx bash start.sh         # 带 Key 启动
#
# 环境变量（全部可选）：
#   变量名             默认值              说明
#   DEEPSEEK_API_KEY   —                   API 密钥（回退）
#   LLM_API_KEY        —                   API 密钥（优先）
#   LLM_MODEL_NAME     deepseek-chat       模型名
#   LLM_API_BASE       https://api.deepseek.com/v1  API 地址
#   LLM_TEMPERATURE    0.7                 温度
#   LLM_MAX_TOKENS     4096                最大 Token 数
#   HOST               0.0.0.0             监听地址
#   PORT               8080                监听端口
#   LOG_LEVEL          info                日志级别
#   CORS_ORIGINS       *                   跨域来源
# ================================================================

set -euo pipefail

cd "$(dirname "$0")"

echo "=== AgentScope3 ==="
echo "目录: $(pwd)"
echo "Python: $(python3 --version 2>&1)"

# 加载 .env（如果存在）
if [ -f ".env" ]; then
    echo "加载 .env 配置"
    # 逐行读取，过滤注释和空行
    while IFS='=' read -r key value || [ -n "$key" ]; do
        key="$(echo "$key" | tr -d '[:space:]')"
        [ -z "$key" ] && continue
        # 跳过注释行
        case "$key" in
            \#*) continue ;;
        esac
        # 仅当环境变量未设置时才从 .env 读取
        if [ -z "${!key:+x}" ]; then
            # 去除首尾空白和引号
            value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')"
            export "$key=$value"
        fi
    done < .env
fi

# 安装依赖（带锁文件，requirements.txt 变更时自动重装）
if [ ! -f ".deps_installed" ] || [ "requirements.txt" -nt ".deps_installed" ]; then
    echo "安装依赖..."
    pip install -r requirements.txt 2>&1 | tail -3
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        touch .deps_installed
        echo "依赖安装完成 ✅"
    else
        echo "依赖安装失败，请检查网络或 requirements.txt" >&2
        exit 1
    fi
fi

# 创建记忆目录
mkdir -p memory

echo ""
echo "启动服务 → ${HOST:-0.0.0.0}:${PORT:-8080}"
echo "模型: ${LLM_MODEL_NAME:-deepseek-chat}"
echo ""

exec python3 server.py
