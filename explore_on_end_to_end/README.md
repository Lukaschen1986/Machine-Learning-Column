# 训推端到端
## 1 - env(2026-05-23)
- 1.创建环境  
conda create -n llm python=3.11.15 -y  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # 必选  
pip install transformers==5.6.0 accelerate==1.7.0  # 必选  
pip install msgpack  # 可选，候补安装  
pip install -U mistral_common  # 可选，候补安装  
  
- 2.验证  
python -c "  
import torch, transformers, accelerate  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0  
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'  
print(f'torch version = {torch.__version__}')  
print(f'cuda version = {torch.version.cuda}')  
print(f'device = {device}; device_count = {device_count}; device_name = {device_name}')  
print(f'transformers version = {transformers.__version__}')  
print(f'accelerate version = {accelerate.__version__}')  
"  
  
## 2 - llama-factory  
- 1.文档地址  
https://llamafactory.readthedocs.io/zh-cn/latest/  
https://github.com/hiyouga/LlamaFactory/blob/main/README_zh.md#%E6%9E%84%E5%BB%BA-docker  
  
- 2.安装  
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git  
cd LlamaFactory  
pip install -e .  
pip install -r requirements/metrics.txt  # 可选  
pip install -r requirements/deepspeed.txt  # 可选，当前只能编译安装  
  
- 3.验证  
llamafactory-cli version  
llamafactory-cli env  
llamafactory-cli webui  # 可选  
  
- 4.训练  
llamafactory-cli train examples/train_full/qwen3_5_full_sft.yaml  # 编辑好配置文件  
nvidia-smi -l 1  # 每秒刷新一次，Ctrl+C 退出  
  
- 5.结论  
成功，但微调后，模型文件与原生存在诸多不一致，待深挖  
  
  
## 3 - transformers serve  
- 1.文档地址  
https://huggingface.co/docs/transformers/main/serving  
  
- 2.安装  
pip install transformers[serving]  
  
- 3.部署  
windows下实测微调后缺失preprocessor_config.json，需手工复制  
scp F:/LLM/Qwen/Qwen3.5-0.8B/preprocessor_config.json C:/my_project/MyGit/LlamaFactory/saves/Qwen3.5-0.8B/full/sft/  
  
设置环境变量  
$env:MODEL_PATH="C:/my_project/MyGit/LlamaFactory/saves/Qwen3.5-0.8B/full/sft"  # windows  
export MODEL_PATH="/LlamaFactory/saves/Qwen3.5-0.8B/full/sft"  # linux  
  
启动服务 - windows  
transformers serve `
$env:MODEL_PATH `
--host "0.0.0.0" `
--port 8080 `
--trust-remote-code `
--dtype bfloat16 `
--device cuda `
--continuous-batching
  
启动服务 - linux  
transformers serve \
$MODEL_PATH \
--host "0.0.0.0" \
--port 8080 \
--trust-remote-code \
--dtype bfloat16 \
--device cuda \
--continuous-batching
  
- 4.测试（微调时指定了qwen3_nothink，因此没有thinking参数）  
curl -X POST http://0.0.0.0:8080/v1/chat/completions \  
  -H "Content-Type: application/json" \  
  -d '{  
    "model": "C:/my_project/MyGit/LlamaFactory/saves/Qwen3.5-0.8B/full/sft",  
    "messages": [  
        {"role": "system", "content": "You are a helpful assistant."},  
        {"role": "user", "content": "你好，你都会什么"}  
    ],  
    "temperature": 0.9,  
    "top_p": 0.9,  
    "max_tokens": 256,  
    "stream": false  
}'  
  
- 5.结论  
成功，原生模型和微调模型均可部署  
  
  
## 4 - llama.cpp  
- 1.文档地址  
https://github.com/ggml-org/llama.cpp  
  
- 2.安装（windows）  
winget install llama.cpp  
  
git@github.com:ggml-org/llama.cpp.git  
cd llama.cpp  
  
python convert_hf_to_gguf.py F:/LLM/Qwen/Qwen3.5-0.8B  
llama-quantize.exe F:/LLM/Qwen/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf F:/LLM/Qwen/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf Q4_K_M  
  
- 3.部署  
$env:MODEL_NAME="F:/LLM/Qwen/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf"  
llama-server.exe `  
-m $env:MODEL_NAME `  
--host "0.0.0.0" `  
--port 8080 `  
-ngl 99 `  
--parallel 4  
  
- 4.测试  
{  
    "model": "Qwen3.5-0.8B-Q4_K_M",  
    "messages": [  
        {"role": "system", "content": "You are a helpful assistant."},  
        {"role": "user", "content": "你好"}  
    ],  
    "temperature": 0.9,  
    "top_p": 0.9,  
    "max_tokens": 256,  
    "stream": false,  
    "chat_template_kwargs": {"enable_thinking": false}  
}  
  
- 5.结论  
成功，但只能部署原生模型，猜测 llama.cpp 不兼容 llama-factory 微调模型  
  
  
## 5 - SGLang (Linux Only)  
https://docs.sglang.io/docs/get-started/install  
  
pip install sglang  


## 6 - vLLM (Linux Only)  
https://docs.vllm.ai/en/stable/getting_started/installation/index.html  
  
pip install vllm --torch-backend=auto  