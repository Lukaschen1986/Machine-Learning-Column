# -*- coding: utf-8 -*-
"""
https://github.com/huggingface/trl
https://huggingface.co/docs/trl/sft_trainer
pip install -U datasets accelerate peft trl tensorboard bitsandbytes langchain sentencepiece --user
pip install transformers==4.37.2 --user
https://github.com/jllllll/bitsandbytes-windows-webui/tree/wheels
"""
import os
import sys
import numpy as np
import pandas as pd
import torch as th
from torch.utils.tensorboard import SummaryWriter
from datasets import (load_dataset, load_from_disk, Dataset)
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, DataCollatorWithPadding, DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq, DataCollatorForTokenClassification)
from transformers.integrations import TensorBoardCallback
from peft import (LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training)
from trl import SFTTrainer


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------
# path
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")
path_log = os.path.join(os.path.dirname(path_project), "log")

# ----------------------------------------------------------------------------------------------------------------
# load dataset official
dataset = load_dataset(
    path="parquet",
    data_files=os.path.join(path_data, "tatsu-lab/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet"),
    split="train"
)
print(dataset)
'''
Dataset({
    features: ['instruction', 'input', 'output', 'text'],
    num_rows: 52002
})
'''

print(dataset[0])  # sft 用 text
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

print(dataset[5])
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

dataset = dataset.select(range(2000))
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=0) 
dataset_train, dataset_test = dataset["train"], dataset["test"]

# ----------------------------------------------------------------------------------------------------------------
# LLM
# checkpoint = "facebook/opt-350m"
# checkpoint = "gpt2"
# checkpoint = "chatglm3-6b"
checkpoint = "Qwen1.5-1.8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
)

tokenizer.pad_token  # '<|endoftext|>'
tokenizer.eos_token  # '<|im_end|>'
# tokenizer.pad_token = tokenizer.eos_token  # 半精度训练时需要
# tokenizer.padding_side = "right"  # llama2
len(tokenizer.get_vocab())  # 151646

config_bnb = BitsAndBytesConfig(
    load_in_8bit=True,
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=th.bfloat16,
    # bnb_4bit_use_double_quant=True
)

model_base = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=th.bfloat16,
    # quantization_config=config_bnb
)

# for param in model_base.parameters():
#     param.requires_grad_(False)

# model_base.is_parallelizable = True
# model_base.model_parallel = True

for i, (name, parm) in enumerate(model_base.named_parameters()):
    print(f"{i}  name: {name};  shape: {parm.shape};  dtype: {parm.dtype};  device: {parm.device}")

print(model_base)
print(model_base.dtype)

# check embedding_size
embedding_size = model_base.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model_base.resize_token_embeddings(len(tokenizer))

# ----------------------------------------------------------------------------------------------------------------
# test base model
# https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat
# https://zhuanlan.zhihu.com/p/690430601
'''
prompt = "程序员有哪些岗位？"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model_base.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

'''

# ----------------------------------------------------------------------------------------------------------------
# model config
config_model = {
    "rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "use_rslora": True,
    "epochs": 10,
    "batch_size": 2,
    "gradient_steps": 2,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "max_seq_lenght": 512
}

# ----------------------------------------------------------------------------------------------------------------
# LoRA
# LoRA: Low-Rank Adaptation of Large Language Models
# config_lora = LoraConfig(target_modules=["0"])
# config_lora = LoraConfig(target_modules=["query_key_value", "dense_4h_to_h"])
# config_lora = LoraConfig(target_modules=[".*\.1.*query_key_value"])
# config_lora = LoraConfig(target_modules=["query_key_value"], modules_to_save=["word_embeddings"])
config_lora = LoraConfig(
    r=config_model.get("rank"),
    lora_alpha=config_model.get("lora_alpha"),
    lora_dropout=config_model.get("lora_dropout"),
    use_rslora=config_model.get("use_rslora"),
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# model_base = prepare_model_for_kbit_training(model_base)
# windows 环境：https://github.com/jllllll/bitsandbytes-windows-webui/tree/wheels
model_lora = get_peft_model(model=model_base, peft_config=config_lora)
model_lora.enable_input_require_grads()  # if TrainingArguments(gradient_checkpointing=True)
model_lora.config.use_cache = False
print(model_lora)
print(config_lora)  # 查看 target_modules


# print_trainable_parameters - 1
model_lora.print_trainable_parameters()

# print_trainable_parameters - 2
trainable_params = 0
all_params = 0

for param in model_lora.parameters():
    if param.requires_grad:
        trainable_params += param.numel()
    all_params += param.numel()

print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")

# ----------------------------------------------------------------------------------------------------------------
# SFT
# train
# args_train = TrainingArguments(
#     output_dir=os.path.join(path_model, "model_sft"),  # 输出目录
#     num_train_epochs=3,  # 训练轮数
#     per_device_train_batch_size=4,  # 训练批次大小
#     per_device_eval_batch_size=4,  # 验证批次大小
#     gradient_accumulation_steps=1,
#     optim="adamw_torch",
#     learning_rate=0.001,  # 0.00005
#     # weight_decay=0.01,
#     warmup_ratio=0.1,
#     lr_scheduler_type="linear",
#     save_strategy="steps",
#     evaluation_strategy="steps",
#     log_level="info",
#     logging_strategy="steps",
#     logging_steps=10,  # 500
#     eval_steps=10,
#     logging_dir=path_log,
#     report_to="all",
#     load_best_model_at_end=False,
#     remove_unused_columns=False,
#     # push_to_hub=False
# )

# args_train = TrainingArguments(
#     output_dir=os.path.join(path_model, "model_sft"),
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     gradient_accumulation_steps=2,  # save mem but waste time
#     gradient_checkpointing=True,    # save mem but waste time
#     optim="adafactor",              # save mem but waste time, paged_adamw_32bit
#     learning_rate=0.001,
#     weight_decay=0.01,
#     logging_strategy="epoch",
#     save_strategy="epoch",
#     evaluation_strategy="epoch",
#     save_total_limit=3,
#     metric_for_best_model="f1",
#     load_best_model_at_end=True
# )

args_train = TrainingArguments(
    output_dir=os.path.join(path_model, "model_sft"),
    num_train_epochs=config_model.get("epochs"),
    per_device_train_batch_size=config_model.get("batch_size"),
    per_device_eval_batch_size=config_model.get("batch_size"),
    gradient_accumulation_steps=config_model.get("gradient_steps"),
    gradient_checkpointing=True, 
    optim="adamw_torch",
    learning_rate=config_model.get("learning_rate"),
    weight_decay=config_model.get("weight_decay"),
    logging_strategy="epoch",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=3,
    #metric_for_best_model="f1",
    load_best_model_at_end=True,
    log_level="info"
)
'''
# 学习曲线
cd ./model_sft
tensorboard --logdir runs
'''

collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)  # 或自定义 collate_fn，参见 demo_4_model.py
# collate_fn = DataCollatorWithPadding(tokenizer)
# collate_fn = DataCollatorForSeq2Seq(tokenizer, padding=True)
# collate_fn = DataCollatorForTokenClassification(tokenizer)
# writer = SummaryWriter()

trainer = SFTTrainer(
    model=model_lora,
    tokenizer=tokenizer,
    args=args_train,
    peft_config=config_lora,
    data_collator=collate_fn,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    dataset_text_field="text",  # 用于指示数据集中哪个字段包含作为模型输入的文本数据
    packing=True,
    max_seq_length=512,
    # callbacks=[TensorBoardCallback(writer)]
    # compute_metrics=compute_metrics,
)

'''
from sklearn.metrics import accuracy_score, f1_score  

def compute_metrics(pred):  
    labels = pred.label_ids  
    preds = pred.predictions.argmax(-1)  
    accuracy = accuracy_score(labels, preds)  
    f1 = f1_score(labels, preds, average='macro')  # 或使用 'micro', 'weighted' 等  
    return {'accuracy': accuracy, 'f1': f1}

def compute_metrics(eval_predict):
    preds, labels = eval_predict
    preds = preds.argmax(axis=-1)
    f1 = f1_score(labels, preds)
    return {"f1": f1}
'''

# model_lora.config.use_cache = False
res_train = trainer.train()
metrics = res_train.metrics
'''
TrainOutput(global_step=18, training_loss=1.8800998263888888, 
            metrics={'train_runtime': 2613.6248, 'train_samples_per_second': 0.028, 
                     'train_steps_per_second': 0.007, 'total_flos': 1322178922414080.0, 
                     'train_loss': 1.8800998263888888, 'epoch': 3.0})
'''
res_eval = trainer.evaluate(dataset_train)
res_eval = trainer.evaluate(dataset_eval)
'''
{'eval_loss': 1.5322265625,
 'eval_runtime': 16.981,
 'eval_samples_per_second': 0.059,
 'eval_steps_per_second': 0.059,
 'epoch': 3.0}
'''

# test
# predictions = trainer.predict(test_dataset=)
# print(predictions.predictions.shape, predictions.label_ids.shape)
# preds = np.argmax(predictions.predictions, axis=-1)
# metric = evaluate.load("glue", "mrpc")
# metric.compute(predictions=preds, references=predictions.label_ids)

# save
trainer.save_model(output_dir=os.path.join(path_model, "model_sft"))

# load
# reload model_base

# load model_sft
model_sft = PeftModel.from_pretrained(
    model=model_base,
    model_id=os.path.join(path_model, "model_sft"),
    is_trainable=False
)
model_sft = model_sft.merge_and_unload()  # W + BA, speed up, but errors when use 8-bit
print(model_sft)

# inference
query = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Design a database to record employee salaries.

### Response:
"""
response, history = model_sft.chat(tokenizer, query=query, history=[])

# save merged model and load
model_sft.save_pretrained(save_directory=os.path.join(path_model, "model_lora"),
                          max_shard_size="1GB")

model_lora = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, "model_lora")
)

# model_lora = th.load(os.path.join(path_model, "model_lora"))
# model_lora.load_state_dict(th.load("..."))
# model_lora.eval()
