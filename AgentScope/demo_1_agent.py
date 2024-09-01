# -*- coding: utf-8 -*-
"""
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple agentscope --pre --user
https://github.com/modelscope/agentscope
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
                                bing_search, create_file, download_from_url)


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
# DialogAgent
agent = DialogAgent(
    name="assistant",
    sys_prompt="You are a helpful assistant.",
    model_config_name=model_config_name
)

msg = Msg(name="Lukas", content="你是谁？", role="user")
res = agent(msg)

print(json.dumps(res, indent=4))
'''
{'id': 'fb8e9bc059914ec5b268bbdc4d0be95b',
 'timestamp': '2024-08-25 11:09:59',
 'name': 'assistant',
 'content': '我是你的助手。有什么问题我可以帮助你解答吗？',
 'role': 'assistant',
 'url': None,
 'metadata': None}
'''

# ----------------------------------------------------------------------------------------------------------------
# DictDialogAgent
agent = DictDialogAgent(
    name="assistant",
    sys_prompt="You are a helpful assistant.",
    model_config_name=model_config_name
)

parser = MarkdownJsonDictParser(
    content_hint={
        "thought": "what you thought",
        "speak": "what you speak",
        # "finish_discussion": "whether the discussion reached an agreement or not (true/false)",
    },
    required_keys=["thought", "speak"],
    keys_to_memory="speak",
    keys_to_content="speak",
    # keys_to_metadata=["finish_discussion"],
)
agent.set_parser(parser)

msg = Msg(name="Lukas", content="你是谁？", role="user")
res = agent(msg)
'''
{'id': 'bb983cbf1f8e4845be1e1198c950d660',
 'timestamp': '2024-08-25 11:12:00',
 'name': 'assistant',
 'content': '我是你的助手，可以提供信息、解答问题或执行任务。你可以问我任何你想要了解的事情！',
 'role': 'assistant',
 'url': None,
 'metadata': None}
'''

# ----------------------------------------------------------------------------------------------------------------
# UserAgent
agent = DialogAgent(
    name="assistant",
    sys_prompt="You are a helpful assistant.",
    model_config_name=model_config_name
)

user = UserAgent(name="user")
msg = user()
res = agent(msg)
'''
{'id': 'f0314da7644240e4a58b36aad415e2b5',
 'timestamp': '2024-08-25 11:35:06',
 'name': 'assistant',
 'content': '我是来自阿里云的大规模语言模型，我叫通义千问。我可以回答各种问题、提供信息、辅助创作和交流等。有什么可以帮助您的吗？',
 'role': 'assistant',
 'url': None,
 'metadata': None}
'''
agent.memory.get_memory()
'''
[{'id': '5f3e3dce5b7346209705c12f9a2a2554',
  'timestamp': '2024-08-25 11:35:05',
  'name': 'user',
  'content': '你是谁？',
  'role': 'user',
  'url': None,
  'metadata': None},
 {'id': 'f0314da7644240e4a58b36aad415e2b5',
  'timestamp': '2024-08-25 11:35:06',
  'name': 'assistant',
  'content': '我是来自阿里云的大规模语言模型，我叫通义千问。我可以回答各种问题、提供信息、辅助创作和交流等。有什么可以帮助您的吗？',
  'role': 'assistant',
  'url': None,
  'metadata': None}]
'''
