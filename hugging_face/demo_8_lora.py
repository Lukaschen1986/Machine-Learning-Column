# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch as th
from torch import nn
import torch.optim as optim
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM)
from peft import (LoraConfig, get_peft_model, PeftModel, TaskType)
from torchkeras import KerasModel


device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# LLM
checkpoint = "chatglm3-6b"  # https://huggingface.co/THUDM/chatglm3-6b

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
    )

pretrained = AutoModel.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
    ).cuda()

model_base = pretrained.eval()

# dct_system_info = {"role": "system",
#                     "content": "你是一个资深导游，擅长为用户制定专业的旅游出行计划。"}
# response, history = model_base.chat(tokenizer, query="你好", history=[dct_system_info])
# response, history = model_base.chat(tokenizer, query="我想去日本，请给我规划一个7天的行程。", history=history)
# response, history = model_base.chat(tokenizer, query="那德国呢？", history=history)

# ----------------------------------------------------------------------------------------------------------------
# PEFT: Parameter-Efficient Fine-Tuning
# LoRA: Low-Rank Adaptation of Large Language Models
# config = LoraConfig(target_modules=["0"], r=8)
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    inference_mode=False
    )
model_lora = get_peft_model(model=model_base, peft_config=config)
model_lora.is_parallelizable = True
model_lora.model_parallel = True
model_lora.print_trainable_parameters()

# ----------------------------------------------------------------------------------------------------------------
# 使用 torchkeras 的 KerasModel
opti = optim.AdamW(params=model_lora.parameters(), lr=0.01, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
estimator = KerasModel(
    net=model_lora,
    loss_fn=None,
    optimizer=opti
    )
ckpt_path = os.path.join(path_model, "model_lora.bin")

# train
estimator.fit(
    train_data=,  # 配置训练数据
    val_data=, # 配置验证数据
    epochs=10,
    patience=5,
    ckpt_path=ckpt_path,
    mixed_precision="fp16",
    plot=True
    )

# valid
model_lora = PeftModel.from_pretrained(model=model_base, model_id=ckpt_path, is_trainable=False)
model_lora = model_lora.merge_and_unload()  # 合并lora权重
model_lora.save_pretrained(save_directory=os.path.join(path_model, "model_lora.bin"),
                           max_shard_size="10G")

# ----------------------------------------------------------------------------------------------------------------
# 使用 transformers 的 trainer
# https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import evaluate

# raw_datasets
raw_datasets = load_dataset("glue", "mrpc")

# tokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# pretrained
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
print(f"参数总量 = {sum(para.nelement() for para in model.parameters())}")

# metrics
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# init args: https://blog.csdn.net/duzm200542901104/article/details/132762582
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")  # 具体查看参数详解

# train
estimator = Trainer(
    data_collator=data_collator,
    tokenizer=tokenizer,
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)
estimator.train()

# valid
predictions = estimator.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)

# save para
estimator.save_model(output_dir="...")

# load para
model.load_state_dict(th.load("..."))
model.eval()
