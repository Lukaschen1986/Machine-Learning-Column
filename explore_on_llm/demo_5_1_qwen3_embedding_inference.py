"""
华为云ModelArts + 昇腾NPU + Qwen3-Embedding-0.6B
基于 sentence_transformers 实现 OpenAI 标准格式 Embedding 推理服务
满足平台MaaS部署要求，支持 float / base64 编码格式

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple
fastapi>=0.100.0
uvicorn>=0.23.0
numpy==1.26.4
torch==2.7.1
transformers==4.51.0
sentence_transformers==5.4.1
pydantic>=2.0.0
datasets>=3.6.0
pyarrow>=15.0.0
dill>=0.3.8
multiprocess>=0.70.16
xxhash>=3.7.0
python-dateutil>=2.8.2
tzdata
"""
import os
import time
import random
import string
import base64
from wsgiref import headers
import numpy as np
from openai import api_key
import torch
import requests
import json

# 设备自动检测：NPU > GPU > CPU
if __import__("importlib").util.find_spec("torch_npu") is not None:
    import torch_npu
    if torch.npu.is_available():
        torch.npu.set_device(0)
        DEVICE = "npu:0"
    else:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"[设备检测] 使用设备：{DEVICE}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer

# ====================== 全局配置 ======================
app = FastAPI(
    title="Qwen3-Embedding OpenAI-Compatible API",
    description="基于sentence_transformers + 昇腾NPU实现的标准OpenAI Embedding服务",
    version="1.0.0"
)

# ModelArts 环境变量（平台自动注入）
MODEL_PATH = os.environ.get("MODEL_PATH", "./Qwen3-Embedding-0.6B")
MAX_SEQ_LENGTH = 512
SERVICE_PORT = 8080

# 全局模型（服务启动时加载一次）
model: SentenceTransformer = None

# ====================== 请求/响应 数据模型 ======================
class EmbeddingRequest(BaseModel):
    """OpenAI 标准请求体"""
    model: str = Field(..., description="模型名称：qwen3-embedding-0.6b")
    input: Union[str, List[str]] = Field(..., description="输入文本：字符串或字符串数组")
    encoding_format: str = Field("float", description="向量编码格式：float / base64")

class EmbeddingData(BaseModel):
    """单条向量结果"""
    index: int
    object: str = "embedding"
    embedding: Union[List[float], str]

class UsageInfo(BaseModel):
    """Token用量统计"""
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int = 0
    prompt_tokens_details: Optional[dict] = None

class EmbeddingResponse(BaseModel):
    """OpenAI 标准响应体"""
    id: str
    object: str = "list"
    created: int
    model: str
    data: List[EmbeddingData]
    usage: UsageInfo

# ====================== 工具函数 ======================
def generate_embedding_id() -> str:
    """生成OpenAI格式的embedding id：embd-随机字符串"""
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=11))
    return f"embd-{suffix}"

def float_vector_to_base64(embedding: List[float]) -> str:
    """
    将float32向量转为base64字符串（OpenAI标准格式）
    """
    np_emb = np.array(embedding, dtype=np.float32)
    return base64.b64encode(np_emb.tobytes()).decode("utf-8")

# ====================== 模型加载（启动钩子） ======================
@app.on_event("startup")
def load_model_on_startup():
    """
    ModelArts 服务启动时加载模型
    仅加载一次，提升推理性能
    """
    global model
    try:
        print(f"[启动加载] 模型路径：{MODEL_PATH}")
        # 使用 sentence_transformers 加载模型（核心要求）
        # 通过 device 参数直接指定设备，避免 .to() 后 SentenceTransformer 内部状态不一致
        model = SentenceTransformer(
            model_name_or_path=MODEL_PATH,
            device=DEVICE,
            trust_remote_code=True,
            cache_folder=MODEL_PATH,
            local_files_only=True
        )
        model.max_seq_length = MAX_SEQ_LENGTH
        print(f"[加载完成] 模型已加载至 {DEVICE}，max_seq_length={MAX_SEQ_LENGTH}")
    except Exception as e:
        print(f"[加载失败] {str(e)}")
        raise RuntimeError(f"模型加载失败：{str(e)}")

# ====================== 核心推理接口 ======================
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest
):
    """
    OpenAI 兼容的 Embedding 推理接口
    - 支持单条/批量文本
    - 支持 float / base64 编码
    - 基于 sentence_transformers 推理
    - NPU 硬件加速
    """
    # 1. 校验模型名称
    if request.model != "qwen3-embedding-0.6b":
        raise HTTPException(
            status_code=400,
            detail=f"仅支持模型：qwen3-embedding-0.6b，传入：{request.model}"
        )

    # 2. 统一输入格式为列表
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    if not texts:
        raise HTTPException(status_code=400, detail="输入内容不能为空")

    # 3. 提前校验编码格式（fail-fast，避免无效请求进入推理阶段）
    if request.encoding_format not in ("float", "base64"):
        raise HTTPException(
            status_code=400,
            detail="encoding_format 仅支持：float、base64"
        )

    # 4. sentence_transformers 核心推理（NPU）
    try:
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                batch_size=len(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,  # 标准Embedding必须归一化
                show_progress_bar=False
            )

        # 4. 统计Token数量（sentence_transformers 统计方式）
        total_tokens = 0
        for text in texts:
            tokens = model.tokenizer.encode(text, truncation=True, max_length=MAX_SEQ_LENGTH)
            total_tokens += len(tokens)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理执行失败：{str(e)}")

    # 5. 构造返回数据
    data_items = []
    for idx, vec in enumerate(embeddings):
        vec_list = vec.tolist()
        result_emb = float_vector_to_base64(vec_list) if request.encoding_format == "base64" else vec_list
        data_items.append(EmbeddingData(index=idx, embedding=result_emb))

    # 6. 返回 OpenAI 标准格式响应
    return EmbeddingResponse(
        id=generate_embedding_id(),
        created=int(time.time()),
        model=request.model,
        data=data_items,
        usage=UsageInfo(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens
        )
    )

# ====================== ModelArts 健康检查 ======================
@app.get("/health")
async def health_check():
    """
    华为云ModelArts必须的健康检查接口
    平台通过该接口判断服务是否正常启动
    """
    return {
        "status": "healthy",
        "model": "qwen3-embedding-0.6b",
        "device": "npu",
        "framework": "sentence_transformers"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level="info"
    )
    '''测试代码
    url = "http://localhost:8080/v1/embeddings"
    api_key = "your_api_key"
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    texts = ["这是一只小猫", "这是一只小狗"]
    
    data = {"model": "qwen3-embedding-0.6b", "input": texts, "encoding_format": "float"}
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    print("Status Code:", response.status_code)
    print("Response:", response.json())
    '''
    