# -*- coding: utf-8 -*-
"""
https://www.volcengine.com/docs/82379/1756990
https://www.volcengine.com/docs/82379/1756990?lang=zh
"""
import warnings; warnings.filterwarnings("ignore")
import os
import requests
import json
import pandas as pd

from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark


# ----------------------------------------------------------------------------------------------------------------
load_dotenv(dotenv_path="vol.env")
API_KEY = os.getenv("API_KEY")

# ----------------------------------------------------------------------------------------------------------------
def inference_with_post(model, system_prompt, user_prompt):
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ]
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.5,
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
    }
    response = requests.post(url, headers=headers, json=data, stream=False)
    
    if response.status_code == 200:
        result = response.json()
        reasoning_content = result["choices"][0]["message"].get("reasoning_content")
        content = result["choices"][0]["message"].get("content")
    else:
        print("请求失败，状态码：", response.status_code)
        print("错误信息：", response.text)
    return reasoning_content, content
    

def inference_with_sdk_base(model, system_prompt, user_prompt):
    url = "https://ark.cn-beijing.volces.com/api/v3"
    client = Ark(base_url=url, api_key=API_KEY)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ]
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        top_p=0.5,
        max_tokens=1024,
        thinking={"type": "disabled"},  # disabled, enabled,
        )
    
    reasoning_content = completion.choices[0].message.reasoning_content
    content = completion.choices[0].message.content    
    return reasoning_content, content
    

def inference_with_sdk_search(model, system_prompt, user_prompt):
    url = "https://ark.cn-beijing.volces.com/api/v3"
    client = Ark(base_url=url, api_key=API_KEY)
    
    curr_time = pd.Timestamp.now()
    suffix = f"\n注：当前时间是 {str(curr_time)}"
    
    instructions = system_prompt
    input = user_prompt + suffix
    
    tools = [{
        "type": "web_search",
        "max_keyword": 3,  # 1-50
        "limit": 10,  # 1-50
        "sources": ["douyin", "moji", "toutiao", "search_engine"],
    }]
    
    # 发起模型请求
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input,
        temperature=0.5,
        top_p=0.5,
        max_output_tokens=1024,
        thinking={"type": "disabled"},  # disabled, enabled,
        tools=tools,
        stream=False,
        )
    
    output = response.output
    query = output[0].action.query
    text = output[1].content[0].text
    annotations = output[1].content[0].annotations
    return query, text, annotations
    
    

if __name__ == "__main__":
    model = "doubao-seed-2-0-pro-260215"
    # model = "doubao-seed-1-6-250615"
    system_prompt = "you are a helpful assistant."
    user_prompt = "3月19号股票大跌的原因是什么？"
    # reasoning_content, content = inference_with_post(model, system_prompt, user_prompt)
    # reasoning_content, content = inference_with_sdk_base(model, system_prompt, user_prompt)
    query, text, annotations = inference_with_sdk_search(model, system_prompt, user_prompt)
    print(text)
    
    



