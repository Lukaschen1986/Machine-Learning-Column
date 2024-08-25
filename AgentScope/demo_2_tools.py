# -*- coding: utf-8 -*-
"""
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple agentscope --pre --user
https://blog.csdn.net/Attitude93/article/details/139263132?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172456653216800186580092%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=172456653216800186580092&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-139263132-null-null.142^v100^pc_search_result_base8&utm_term=agentscope%E5%B7%A5%E5%85%B7&spm=1018.2226.3001.4187
https://blog.csdn.net/Attitude93/article/details/139263305?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172456653216800186580092%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=172456653216800186580092&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-139263305-null-null.142^v100^pc_search_result_base8&utm_term=agentscope%E5%B7%A5%E5%85%B7&spm=1018.2226.3001.4187
https://blog.csdn.net/Attitude93/article/details/139263146
"""
import warnings; warnings.filterwarnings("ignore")
import os
import sys
from typing import Any
import json
import numpy as np
import pandas as pd
import torch as th
import agentscope
from agentscope import msghub
from agentscope.agents import (UserAgent, DialogAgent, DictDialogAgent, ReActAgent)
from agentscope.parsers.json_object_parser import MarkdownJsonDictParser
from agentscope.message import Msg
from agentscope.service import (ServiceExecStatus, ServiceToolkit, ServiceResponse, 
                                bing_search, create_file, download_from_url,
                                summarization)


device = th.device("cuda" if th.cuda.is_available() else "cpu")
devive_cnt = th.cuda.device_count()
print(f"device = {device}; devive_cnt = {devive_cnt}")
print(th.__version__)
print(th.version.cuda)
print(agentscope.__version__)

# ----------------------------------------------------------------------------------------------------------------
# config & init
model_configs = [
    {
        "config_name": "ollama_chat-qwen2:7b",
        "model_type": "ollama_chat",
        "model_name": "qwen2:7b",
        "options": {"temperature": 1.0}
    },
    {
        "config_name": "ollama_chat-qwen:14b",
        "model_type": "ollama_chat",
        "model_name": "qwen:14b",
        "options": {"temperature": 1.0}
    },
    {
        "config_name": "ollama_chat-qwen:32b",
        "model_type": "ollama_chat",
        "model_name": "qwen:32b",
        "options": {"temperature": 1.0}
    }
    ]

agentscope.init(model_configs)
model_config_name = "ollama_chat-qwen2:7b"
# model_config_name = "ollama_chat-qwen:32b"
# model_config_name = "ollama_chat-qwen:14b"

# ----------------------------------------------------------------------------------------------------------------
# bing_search
service_toolkit = ServiceToolkit()

service_toolkit.add(
    bing_search,
    api_key="xxx",
    num_results=3
)

print(service_toolkit.tools_instruction)
print(service_toolkit.json_schemas)
print(service_toolkit.tools_calling_format)

# 当输入为字符串时，此函数将相应地解析字符串并使用解析后的参数执行函数。
string_input = '[{"name": "bing_search", "arguments": {"question": "xxx"}}]'
res_of_string_input = service_toolkit.parse_and_call_func(string_input)

# 而如果输入为解析后的字典，则直接调用函数。
dict_input = [{"name": "bing_search", "arguments": {"question": "xxx"}}]
res_of_dict_input = service_toolkit.parse_and_call_func(dict_input)

# ----------------------------------------------------------------------------------------------------------------
# download_from_url
service_toolkit = ServiceToolkit()
service_toolkit.add(service_func=download_from_url)

url = "https://img2023.cnblogs.com/blog/719190/202308/719190-20230829160824073-46349672.png"
filepath = "C:/my_project/MyGit/Machine-Learning-Column/AgentScope/sample.png"
dict_input = [{"name": "download_from_url", "arguments": {"url": url, "filepath": filepath}}]
res_of_dict_input = service_toolkit.parse_and_call_func(dict_input)

# ----------------------------------------------------------------------------------------------------------------
# 自定义工具
def sum_num(a: int, b: int) -> ServiceResponse:
    """计算两个数的和

    Args:
        a (int): 参数1
        b (int): 参数2

    Returns:
        int: 结果
    """
    output = a + b
    status = ServiceExecStatus.SUCCESS
    return ServiceResponse(status, output)

service_toolkit = ServiceToolkit()
service_toolkit.add(sum_num)
print(service_toolkit.service_funcs)  # dict

content = "1加1，然后加3，最后再乘以2，等于几？"
msg = Msg(name="user", content=content, role="user")

# ReActAgent, https://arxiv.org/abs/2210.03629
agent_react = ReActAgent(
    name="assistant",
    model_config_name=model_config_name,
    service_toolkit=service_toolkit,
    max_iters=3
)
print(agent_react.sys_prompt)
res = agent_react(msg)

# ----------------------------------------------------------------------------------------------------------------
# 自定义工具
def air(departName: str, arriveName: str) -> ServiceResponse:
    """
    预订机票的工具
    
    Args:
        departName (`str`): 出发城市名称
        arriveName (`str`): 到达城市名称
    
    Returns:
        `ServiceResponse`: If the model successfully get the air lines, return
        `ServiceResponse` with `ServiceExecStatus.SUCCESS`; otherwise return
        `ServiceResponse` with `ServiceExecStatus.ERROR`.
    """
    res = f"正在为您查找从{departName}到{arriveName}的机票，请您耐心等待..."
    status = ServiceExecStatus.SUCCESS
    return ServiceResponse(status, res)

def hotel(cityName: str) -> ServiceResponse:
    """
    预订酒店的工具
    
    Args:
        cityName (`str`): 城市名称
    
    Returns:
        `ServiceResponse`: If the model successfully get the hotel list, return
        `ServiceResponse` with `ServiceExecStatus.SUCCESS`; otherwise return
        `ServiceResponse` with `ServiceExecStatus.ERROR`.
    """
    res = f"正在为您查找{cityName}的酒店信息，请您耐心等待..."
    status = ServiceExecStatus.SUCCESS
    return ServiceResponse(status, res)

service_toolkit = ServiceToolkit()
service_toolkit.add(air)
service_toolkit.add(hotel)

content = "我要去深圳出差，从南京出发，请帮我预订机票。"
# content = "我要去深圳出差，请帮我预订酒店。"
# content = "我要去深圳出差，从南京出发，请帮我预订机票，然后再订酒店。"  # 定义一个整合工具
msg = Msg(name="user", content=content, role="user")

agent_react = ReActAgent(
    name="assistant",
    model_config_name=model_config_name,
    service_toolkit=service_toolkit,
    max_iters=3
)
res = agent_react(msg)
res
