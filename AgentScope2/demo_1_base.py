# -*- coding: utf-8 -*-
"""
AgentScope 文档：
https://doc.agentscope.io/zh_CN/tutorial/quickstart_installation.html

火山方舟控制台：
https://console.volcengine.com/ark/region:ark+cn-beijing/overview?briefPage=0&briefType=introduce&type=new
"""
import warnings; warnings.filterwarnings("ignore")
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
from agentscope.formatter import OpenAIChatFormatter
from agentscope.tool import (Toolkit, ToolResponse, execute_python_code)
from agentscope.agent import ReActAgent
from agentscope.message import (TextBlock, Msg)


print(f"AgentScope version: {agentscope.__version__}")
print(f"OpenAI version: {openai.__version__}")

# ----------------------------------------------------------------------------------------------------------------------
curr_path = "C:\\my_project\\MyGit\\Machine-Learning-Column\\AgentScope2"
load_dotenv(dotenv_path=os.path.join(curr_path, "agent.env"))
API_KEY = os.getenv("VOL_API_KEY")

nest_asyncio.apply()  # 解决 asyncio.run() 在 Jupyter Notebook 中的 RuntimeError: This event loop is already running. 问题

# ----------------------------------------------------------------------------------------------------------------------
# 定义一个异步函数来调用聊天模型
async def chat_model(model_name: str, user_prompt: str) -> str:
    model = OpenAIChatModel(
        model_name=model_name,
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
        stream=False,
    )
    
    message = [
        {"role": "user", "content": user_prompt}
    ]
    
    res = await model(message)
    return res.content[0]["text"] # type: ignore

# AgentScope 中的流式返回结果为累加式，这意味着每个 chunk 中的内容包含所有之前的内容加上新生成的内容。
async def chat_model_stream(model_name: str, user_prompt: str) -> None:
    model = OpenAIChatModel(
        model_name=model_name,
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
    
    message = [
        {"role": "user", "content": user_prompt}
    ]
    
    generator = await model(message)
    async for chunk in generator: # type: ignore
        print(f"\t类型: {type(chunk.content)}")
        print(f"\t{chunk}\n")
    return 


if __name__ == "__main__":
    # model_name="doubao-seed-2-0-pro-260215"
    model_name="deepseek-v3-2-251201"
    user_prompt = "你好，朋友"
    
    text = asyncio.run(chat_model(model_name, user_prompt))   # 非流式
    asyncio.run(chat_model_stream(model_name, user_prompt))   # 流式
