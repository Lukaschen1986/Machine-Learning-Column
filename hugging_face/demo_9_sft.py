# -*- coding: utf-8 -*-
"""
https://github.com/huggingface/trl
https://huggingface.co/docs/trl/sft_trainer
pip install -U datasets transformers accelerate peft trl --user
pip install -U bitsandbytes --user
https://github.com/jllllll/bitsandbytes-windows-webui/tree/wheels
"""
import os
import sys
import numpy as np
import torch as th
from datasets import (load_dataset, load_from_disk, Dataset)
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling)
from peft import (LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_int8_training)
from trl import SFTTrainer


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------
# path
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# load dataset
# https://huggingface.co/datasets/tatsu-lab/alpaca
dataset_train = load_dataset(
    path="parquet",
    data_files=os.path.join(path_data, "tatsu-lab/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet"),
    split="train"
    )
print(dataset_train)
'''
Dataset({
    features: ['instruction', 'input', 'output', 'text'],
    num_rows: 52002
})
'''

print(dataset_train[0])  # sft 用 text
'''
{
 'instruction': 'Give three tips for staying healthy.', 
 'input': '', 
 'output': (
     '1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n'
     '2. Exercise regularly to keep your body active and strong. \n'
     '3. Get enough sleep and maintain a consistent sleep schedule.'
     ), 
 'text': (
      'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
      '### Instruction:\nGive three tips for staying healthy.\n\n'
      '### Response:\n1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n'
      '2. Exercise regularly to keep your body active and strong. \n'
      '3. Get enough sleep and maintain a consistent sleep schedule.'
      )
      }
'''

print(dataset_train[5])
'''
{
 'instruction': 'Identify the odd one out.', 
 'input': 'Twitter, Instagram, Telegram', 
 'output': 'Telegram', 
 'text': (
     'Below is an instruction that describes a task, paired with an input that provides further context. '
     'Write a response that appropriately completes the request.\n\n'
     '### Instruction:\nIdentify the odd one out.\n\n'
     '### Input:\nTwitter, Instagram, Telegram\n\n'
     '### Response:\nTelegram')}
'''

# ----------------------------------------------------------------------------------------------------------------
# LLM
# https://huggingface.co/Salesforce/xgen-7b-8k-base
# https://huggingface.co/facebook/opt-350m
# https://huggingface.co/THUDM/chatglm3-6b
checkpoint = "facebook/opt-350m"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
    )

tokenizer.pad_token  # '<pad>'
tokenizer.eos_token  # '</s>'
# tokenizer.pad_token = tokenizer.eos_token
len(tokenizer.get_vocab())  # 50265

config_bnb = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=False,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=th.bfloat16
    )

pretrained = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True,
    device_map=device,
    torch_dtype=th.bfloat16,
    # quantization_config=config_bnb
    )
model_L1 = pretrained.eval()

for param in model_L1.parameters():
    param.requires_grad_(False)

for i, (name, parm) in enumerate(model_L1.named_parameters()):
    print(f"{i}  name: {name};  shape: {parm.shape};  dtype: {parm.dtype};  device: {parm.device}")
'''
0  name: model.decoder.embed_tokens.weight;  shape: torch.Size([50272, 512]);  dtype: torch.bfloat16;  device: cuda:0
1  name: model.decoder.embed_positions.weight;  shape: torch.Size([2050, 1024]);  dtype: torch.bfloat16;  device: cuda:0
2  name: model.decoder.project_out.weight;  shape: torch.Size([512, 1024]);  dtype: torch.bfloat16;  device: cuda:0

385  name: model.decoder.layers.23.fc2.bias;  shape: torch.Size([1024]);  dtype: torch.bfloat16;  device: cuda:0
386  name: model.decoder.layers.23.final_layer_norm.weight;  shape: torch.Size([1024]);  dtype: torch.bfloat16;  device: cuda:0
387  name: model.decoder.layers.23.final_layer_norm.bias;  shape: torch.Size([1024]);  dtype: torch.bfloat16;  device: cuda:0
'''

print(model_L1)

# 非必须
model_L1.resize_token_embeddings(len(tokenizer))  # Embedding(50265, 512)

# ----------------------------------------------------------------------------------------------------------------
# LoRA
# LoRA: Low-Rank Adaptation of Large Language Models
# config_lora = LoraConfig(target_modules=["0"], r=8)
config_lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
    )

model_L2 = prepare_model_for_int8_training(model_L1)
model_L2 = get_peft_model(model=model_L1, peft_config=config_lora)  # windows 环境：https://github.com/jllllll/bitsandbytes-windows-webui/tree/wheels
print(model_L2)

# model_L2.is_parallelizable = True
# model_L2.model_parallel = True

# print_trainable_parameters - 1
model_L2.print_trainable_parameters()

# print_trainable_parameters - 2
trainable_params = 0
all_params = 0
for (_, param) in model_L2.named_parameters():
    if param.requires_grad:
        trainable_params += param.numel()
    all_params += param.numel()

print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")

# ----------------------------------------------------------------------------------------------------------------
# SFT
# train
args_train = TrainingArguments(
    output_dir=os.path.join(path_model, "model_L2"),
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    learning_rate=0.00005,
    # weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    evaluation_strategy="epoch",
    logging_steps=10,
    # fp16=True,
    # load_best_model_at_end=True,
    # push_to_hub=False
    )

collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # 或自定义 collate_fn，参见 demo_4_model.py

estimator = SFTTrainer(
    model=model_L2,
    tokenizer=tokenizer,
    args=args_train,
    peft_config=config_lora,
    data_collator=collate_fn,
    train_dataset=dataset_train,
    # eval_dataset=,
    dataset_text_field="text",
    packing=True,
    max_seq_length=512,
    # compute_metrics=,
    )

# model_L2.config.use_cache = False
estimator.train()

# valid
predictions = estimator.predict(test_dataset=)
print(predictions.predictions.shape, predictions.label_ids.shape)

preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)

# save para
estimator.save_model(output_dir="...")

# load para
model.load_state_dict(th.load("..."))
model.eval()






