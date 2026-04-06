# -*- coding: utf-8 -*-
"""
章节文档：
https://doc.agentscope.io/zh_CN/tutorial/workflow_conversation.html
"""
from re import U
import warnings; warnings.filterwarnings("ignore")
import sys
import urllib3
import os
import requests
import json
import agentscope
import openai
import httpx
import asyncio
import nest_asyncio

from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import (BaseModel, Field)
from agentscope.model import (OpenAIChatModel, ChatResponse)
from agentscope.memory import InMemoryMemory
from agentscope.formatter import (OpenAIChatFormatter, OpenAIMultiAgentFormatter)
from agentscope.tool import (Toolkit, ToolResponse, execute_python_code)
from agentscope.agent import (ReActAgent, AgentBase, UserAgent)
from agentscope.message import (TextBlock, Msg)
from agentscope.pipeline import MsgHub


print(f"AgentScope version: {agentscope.__version__}")
print(f"OpenAI version: {openai.__version__}")
"""
AgentScope version: 1.0.18
OpenAI version: 2.30.0
"""

# ----------------------------------------------------------------------------------------------------------------------
# 设置 API Key 和其他配置
curr_path = "C:\\my_project\\MyGit\\Machine-Learning-Column\\AgentScope2"
load_dotenv(dotenv_path=os.path.join(curr_path, "agent.env"))
API_KEY = os.getenv("VOL_API_KEY")

nest_asyncio.apply()  # 解决 asyncio.run() 在 Jupyter Notebook 中的 RuntimeError: This event loop is already running. 问题

# ----------------------------------------------------------------------------------------------------------------------
# User-Assistant 对话
# 创建一个工具箱，并注册一个工具函数
toolkit = Toolkit()

# 创建一个内存对象
memory = InMemoryMemory()

# 创建一个 ReActAgent 实例
jarvis = ReActAgent(
    name="Jarvis",
    sys_prompt="你是一个名为 Jarvis 的助手，始终以中文与用户对话。",
    model=OpenAIChatModel(
        model_name="deepseek-v3-2-251201",
        api_key=API_KEY,
        client_kwargs={
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "http_client": httpx.AsyncClient(verify=False),
        }, # type: ignore
        generate_kwargs={
            "temperature": 0.5,
            "top_p": 0.5,
            "max_tokens": 512,
        },
        stream=True,
    ),
    formatter=OpenAIChatFormatter(),
    toolkit=toolkit,
    memory=memory,
)

# 创建用户智能体
user = UserAgent(name="User")

# 我们可以通过在这两个智能体之间交换消息来开始对话
async def run_conversation() -> None:
    """运行 Jarvis 和用户之间的简单对话。"""
    msg = None
    while True:
        msg = await jarvis(msg)
        msg = await user(msg)
        if msg.get_text_content() == "再见":
            print("对话结束。")
            break


if __name__ == "__main__":
    asyncio.run(run_conversation())
 