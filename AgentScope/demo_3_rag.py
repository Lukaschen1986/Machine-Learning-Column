# -*- coding: utf-8 -*-
"""
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple agentscope --pre --user
https://blog.csdn.net/Attitude93/article/details/139448187
https://blog.csdn.net/Attitude93/article/details/139452945
https://blog.csdn.net/Attitude93/article/details/139478271
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


