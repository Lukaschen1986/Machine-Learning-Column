"""
Qwen3.5-9B 全参数微调（DDP + DeepSpeed）

基于 demo_2_2_training_for_fft.ipynb 改写：
  - 移除 device_map="auto"（DDP/DeepSpeed 不兼容）
  - 移除 model_parallel / is_parallelizable（那是模型并行用的）
  - 新增 deepspeed 参数，加载 ZeRO 配置文件
  - 新增 local_rank 判据，只从 rank 0 保存
  - 支持 torch_npu（昇腾 910B3）

用法：
  # 单机 8 卡（你的场景）
  torchrun --nproc_per_node=8 train_fft_ddp.py \
      --model_path /path/to/Qwen3.5-9B-Instruct \
      --data_path /path/to/data.json \
      --output_dir /path/to/output \
      --deepspeed_config ds_zero3.json

  # 两机 16 卡（ModelArts 会自动处理 master_addr）
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$RANK \
      --master_addr=$MASTER_ADDR --master_port=29500 \
      train_fft_ddp.py \
      --model_path /path/to/Qwen3.5-9B-Instruct \
      --data_path /path/to/data.json \
      --output_dir /path/to/output \
      --deepspeed_config ds_zero3.json
"""

import os
import sys
import warnings
import argparse

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pprint import pp

import torch as th
import transformers
import accelerate
import trl
import peft

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTConfig, SFTTrainer

# ── 解析命令行参数 ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=None,
                    help="模型路径，如 /path/to/Qwen3.5-9B-Instruct")
parser.add_argument("--data_path", type=str, default=None,
                    help="训练数据路径，如 /path/to/train_data.json")
parser.add_argument("--output_dir", type=str, default=None,
                    help="输出目录，如 /path/to/output")
parser.add_argument("--deepspeed_config", type=str, default=None,
                    help="DeepSpeed 配置文件路径")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="分布式 local_rank（torchrun 自动传入）")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=1,
                    help="per_device_train_batch_size（显存紧张时设 1）")
parser.add_argument("--grad_accum", type=int, default=4,
                    help="gradient_accumulation_steps")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--warmup_steps", type=float, default=0.03)
parser.add_argument("--packing", action="store_true", default=False,
                    help="启用 packing（需 flash_attention_2）")
parser.add_argument("--save_total_limit", type=int, default=2)
args = parser.parse_args()

# ── 分布式环境初始化 ───────────────────────────────────────
# 如果没有传 --local_rank，从环境变量拿（torchrun 自动设置）
local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))

is_main_process = (rank == 0)  # 只在主进程输出日志 / 保存模型

# ── NPU / CUDA 自适应 ──────────────────────────────────────
if th.cuda.is_available():
    device_type = "cuda"
    device = th.device(f"cuda:{local_rank}" if local_rank >= 0 else "cuda")
elif sys.platform == "linux":
    # 适配昇腾 910B3（ModelArts 镜像自带 torch_npu）
    import torch_npu
    if th.npu.is_available():
        device_type = "npu"
        device = th.device(f"npu:{local_rank}" if local_rank >= 0 else "npu")
    else:
        device = th.device("cpu")
        device_type = "cpu"
else:
    device = th.device("cpu")
    device_type = "cpu"

if is_main_process:
    print(f"⚙️  torch version    = {th.__version__}")
    print(f"⚙️  transformers     = {transformers.__version__}")
    print(f"⚙️  accelerate       = {accelerate.__version__}")
    print(f"⚙️  trl              = {trl.__version__}")
    print(f"⚙️  peft             = {peft.__version__}")
    print(f"⚙️  device_type      = {device_type}")
    print(f"⚙️  world_size       = {world_size}")
    print(f"⚙️  local_rank       = {local_rank}")
    print(f"⚙️  deepspeed_cfg    = {args.deepspeed_config}")


# ============================================================
#  Step-1: 数据源
# ============================================================
if is_main_process:
    print("\n📥 加载数据...")

dataset = load_dataset(
    path="json",
    data_files=args.data_path,
    split="all",
)

# 切分训练 / 验证（10% 验证）
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=0)
train_dataset, eval_dataset = dataset["train"], dataset["test"]

if is_main_process:
    print(f"📦 训练集: {len(train_dataset)} 条 | 验证集: {len(eval_dataset)} 条")
    if len(train_dataset) > 0:
        pp(train_dataset[0])


# ============================================================
#  Step-2: Tokenizer
# ============================================================
if is_main_process:
    print("\n🔤 加载 tokenizer...")

# ── 改动：如果没传 --model_path，从环境变量 fallback ──
model_path = args.model_path or os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-0.8B")

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if is_main_process:
        print("⚠️  pad_token 为 None，已设为 eos_token")

if is_main_process:
    print(f"🔤  eos_token     = {tokenizer.eos_token}")
    print(f"🔤  pad_token     = {tokenizer.pad_token}")
    print(f"🔤  padding_side  = {tokenizer.padding_side}")


# ============================================================
#  Step-3: 量化（跳过，全参数微调不需要）
# ============================================================


# ============================================================
#  Step-4: 载入基模 ⚠️ 关键改动
# ============================================================
if is_main_process:
    print("\n🧠 加载基模...")

# ── 改动 1：移除 device_map="auto"
#     DDP / DeepSpeed 下每个进程负责一张卡，device_map 会冲突
# ── 改动 2：移除 low_cpu_mem_usage（DeepSpeed 自己处理）
base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=th.bfloat16,           # 910B3 原生支持 BF16
    attn_implementation="sdpa",        # 昇腾兼容，不用 flash_attention_2
    # quantization_config=...,         # 全参数微调不要量化
)

# ── 改动 3：tokenizer > embedding size 时 resize
tokenizer_size = len(tokenizer)
embedding_size = base_model.config.vocab_size
if tokenizer_size > embedding_size:
    base_model.resize_token_embeddings(tokenizer_size)
    if is_main_process:
        print(f"🔧  embedding 已从 {embedding_size} resize 到 {tokenizer_size}")

# ── 改动 4：去掉 is_parallelizable / model_parallel
#     那些是 transformers 的模型并行（tensor/pipeline parallelism）用的，
#     DDP / DeepSpeed 不需要
base_model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
base_model.enable_input_require_grads()
base_model.config.use_cache = False  # 训练时必须关

if is_main_process:
    # 打印模型参数概况
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"📊  总参数量: {total_params / 1e9:.2f}B")
    print(f"📊  可训练:   {trainable_params / 1e9:.2f}B")


# ============================================================
#  Step-5: 训练参数
# ============================================================
if is_main_process:
    print("\n⚙️  配置训练参数...")

output_dir = args.output_dir or os.environ.get("OUTPUT_DIR", "./output_model_fft")

train_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    gradient_checkpointing=True,
    optim="adamw_torch",
    learning_rate=args.lr,
    warmup_ratio=args.warmup_steps,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    dataset_text_field="text",
    packing=args.packing,
    save_total_limit=args.save_total_limit,
    save_only_model=True,

    # ── 改动 5：DeepSpeed 配置 ──
    deepspeed=args.deepspeed_config,

    # ── 改动 6：DDP 参数 ──
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=25,

    # ── 改动 7：NPU 环境不需数据并行后端 ──
    #     DeepSpeed 自动处理
)

if is_main_process:
    print(f"📁  输出目录: {output_dir}")
    print(f"📁  batch_size: {args.batch_size}, grad_accum: {args.grad_accum}")
    print(f"📁  effective_batch_size: {args.batch_size * args.grad_accum * world_size}")
    print(f"📁  deepspeed: {args.deepspeed_config}")


# ============================================================
#  Step-6: LoRA（全参数微调，跳过）
# ============================================================


# ============================================================
#  Step-7: 整理函数（数据格式转换）
# ============================================================
system_prompt = "You are a helpful assistant."


def apply_sft_template(sample):
    """数据集整理函数 → chat template 格式

    Alpaca 结构: instruction + input → output
    如果数据格式不同，按需修改此函数
    """
    user_prompt = sample["instruction"]
    if sample.get("input", "").strip():
        user_prompt += "\n" + sample["input"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": sample["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  # Qwen3.5 默认开启思考，这里显式关闭
    )
    sample["text"] = text
    return sample


if is_main_process:
    print("\n🔄  应用 chat template...")

dataset = dataset.map(apply_sft_template)

if is_main_process:
    print("✅  模板应用完成")
    # 展示一条示例
    pp(dataset["train"][0]["text"][:200])


# ============================================================
#  Step-8: Trainer
# ============================================================
if is_main_process:
    print("\n🏋️  初始化 Trainer...")

trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    args=train_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
)

# ── 改动 8：打印可训练参数（确认全参数微调）──
if is_main_process:
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"🎯  可训练参数: {trainable / 1e9:.2f}B / {total / 1e9:.2f}B")


# ============================================================
#  Step-9: 开始训练
# ============================================================
if is_main_process:
    print("\n🚀  开始训练...（仅主进程输出进度）")

trainer.train()

# ── 改动 9：只从 rank 0 保存模型 ──
#     DeepSpeed ZeRO-3 下，只有 rank 0 持有完整的权重。
#     save_model() 内部会等所有 rank 同步，非主进程不额外操作。
if is_main_process:
    print("\n💾  保存模型...（仅 rank 0）")
    trainer.save_model(output_dir=os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"✅  模型已保存到: {os.path.join(output_dir, 'final')}")

if is_main_process:
    print("\n🎉  训练完成！")


# ============================================================
#  Step-10: 推理验证（可选，仅 rank 0 执行）
# ============================================================
if is_main_process and len(train_dataset) > 0:
    print("\n🧪  推理验证...")
    sample = train_dataset[0]
    user_prompt = sample["instruction"]
    if sample.get("input", "").strip():
        user_prompt += "\n" + sample["input"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    base_model.eval()
    with th.inference_mode():
        complete_ids = base_model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=1024,
            top_p=0.5,
            temperature=0.5,
            do_sample=True,
        )
    response = tokenizer.decode(
        complete_ids[0][model_inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    print(f"📝  输入: {user_prompt[:100]}...")
    print(f"📝  输出: {response[:200]}...")
