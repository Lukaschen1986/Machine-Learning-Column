"""
https://github.com/Lukaschen1986/evalscope/blob/main/README_zh.md
https://evalscope.readthedocs.io/zh-cn/latest/
https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html
"""
import warnings; warnings.filterwarnings("ignore")
import os
import sys

from dotenv import load_dotenv
from evalscope import (TaskConfig, run_task)
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.arguments import Arguments


# ----------------------------------------------------------------------------------------------------------------------
# path info
path_project = os.getcwd()
load_dotenv(dotenv_path=os.path.join(path_project, "evalscope.env"))
api_key = os.getenv("...")

# ----------------------------------------------------------------------------------------------------------------------
# 基本应用
task_cfg_eval = TaskConfig(
    model="qwen3:8b",
    api_url="http://localhost:11434/v1/chat/completions",
    api_key="EMPTY",
    eval_type="service",
    datasets=["chinese_simpleqa"],
    dataset_args={
        "chinese_simpleqa": {"local_path": os.path.join(path_project, "data/chinese_simpleqa")}
    },
    eval_batch_size=2,
    generation_config={
        "max_token": 128,
        "temperature": 0.5,
        "top_p": 0.5
    },
    stream=True,
    timeout=60000,
    limit=10,
)

# ----------------------------------------------------------------------------------------------------------------------
# 压测
task_cfg_perf = Arguments(
    parallel=[1, 5, 10, 20],  # 并发数，同时发送请求的客户端数量
    number=[10, 10, 10, 10],  # 每个并发的总数量，要与parallel一一对应
    model="qwen3:8b",
    url="http://localhost:11434/v1/chat/completions",
    api="openai",
    api_key="EMPTY",
    dataset="random",
    min_prompt_length=128,
    max_prompt_length=1024,
    tokenizer_path="F:/LLM/Qwen/Qwen3-8B",
    # extra_args={"ignore_eos": True},
)



if __name__ == "__main__":
    # results = run_task(task_cfg_eval)
    results = run_perf_benchmark(task_cfg_perf)