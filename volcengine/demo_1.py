# -*- coding: utf-8 -*-
"""
https://www.volcengine.com/docs/82379/1756990
"""
import warnings; warnings.filterwarnings("ignore")
import os
# import json
# import requests

from dotenv import load_dotenv
from openai import OpenAI
from volcenginesdkarkruntime import Ark


# ----------------------------------------------------------------------------------------------------------------
load_dotenv(dotenv_path="vol.env")
vol_key = os.getenv("VOL_KEY")

# ----------------------------------------------------------------------------------------------------------------
client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    api_key=vol_key
    )

# ----------------------------------------------------------------------------------------------------------------
model = "doubao-seed-1-6-250615"

system_prompt = """you are a helpful assistant."""
user_prompt = "你好呀"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
    ]

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    thinking={
        "type": "disabled", # 不使用深度思考能力
          # "type": "enabled", # 使用深度思考能力
          # "type": "auto", # 模型自行判断是否使用深度思考能力
        },
    max_completion_tokens=1024,
    )

print(completion.choices[0].message)

# ----------------------------------------------------------------------------------------------------------------
client = OpenAI(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=vol_key
)

tools = [{
    "type": "web_search",
}]

system_prompt = """you are a helpful assistant."""
user_prompt = "北京的天气怎么样？"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
    ]

# 创建一个对话请求
response = client.responses.create(
    model=model,
    input=messages,
    tools=tools,
)

print(response)


