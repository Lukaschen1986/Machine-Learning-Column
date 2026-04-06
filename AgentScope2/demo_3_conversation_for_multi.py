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
# 多实体对话
toolkit = Toolkit()
memory = InMemoryMemory()
formatter = OpenAIMultiAgentFormatter()

model = OpenAIChatModel(
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
)

alice = ReActAgent(
    name="Alice",
    sys_prompt="你是一个名为 Alice 的学生。",
    model=model,
    formatter=formatter,
    toolkit=toolkit,
    memory=memory,
)

bob = ReActAgent(
    name="Bob",
    sys_prompt="你是一个名为 Bob 的学生。",
    model=model,
    formatter=formatter,
    toolkit=toolkit,
    memory=memory,
)

charlie = ReActAgent(
    name="Charlie",
    sys_prompt="你是一个名为 Charlie 的学生。",
    model=model,
    formatter=formatter,
    toolkit=toolkit,
    memory=memory,
)

async def run_msghub() -> None:
    """运行 MsgHub 来协调 Alice、Bob 和 Charlie 之间的对话。"""
    async with MsgHub(
        participants=[alice, bob, charlie],
        announcement=Msg(
            name="System",
            content="现在大家互相认识一下，简单自我介绍。",
            role="system",
        ),
    ) as hub:
        await alice()
        await bob()
        await charlie()
    
    print("Alice 的记忆：")
    for msg in await alice.memory.get_memory():
        print(f"{msg.name}: {json.dumps(msg.content, indent=4, ensure_ascii=False)}")
    

if __name__ == "__main__":
    asyncio.run(run_msghub())
    
    