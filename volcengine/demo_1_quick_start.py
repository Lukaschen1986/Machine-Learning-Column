# -*- coding: utf-8 -*-
"""
https://www.volcengine.com/docs/82379/1756990
"""
import warnings; warnings.filterwarnings("ignore")
import os

from dotenv import load_dotenv
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
        "type": "auto",  # disabled, enabled
        },
    max_completion_tokens=1024,
    )

print(completion.choices[0].message)


