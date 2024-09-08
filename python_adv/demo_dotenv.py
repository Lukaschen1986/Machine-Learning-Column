# -*- coding: utf-8 -*-
"""
pip install python-dotenv
"""
import os
from dotenv import load_dotenv


# 全局
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY")
print(hf_api_key)  # your_key

# 项目
load_dotenv(dotenv_path="py_adv.env")
hf_api_key = os.getenv("HF_API_KEY")
print(hf_api_key)  # your_key_in_py_adv

