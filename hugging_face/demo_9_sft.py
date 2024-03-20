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
from peft import (LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_int8_training)
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
# load dataset unofficial
lst_train = []
lst_eval = []
idx_train = 100
idx_eval = 10

for i in range(idx_train):
    text = dataset_train[i]["text"]
    lst_train.append(text)

for i in range(idx_train, idx_train + idx_eval):
    text = dataset_train[i]["text"]
    lst_eval.append(text)

dataset_train_2 = pd.DataFrame(data=lst_train, columns=["text"])
dataset_eval_2 = pd.DataFrame(data=lst_eval, columns=["text"])

dataset_train_2.to_parquet(os.path.join(path_data, "tatsu-lab/alpaca/dataset_train_2.parquet"))
dataset_eval_2.to_parquet(os.path.join(path_data, "tatsu-lab/alpaca/dataset_eval_2.parquet"))

dataset_train = load_dataset(
    path="parquet",
    data_files=os.path.join(path_data, "tatsu-lab/alpaca/dataset_train_2.parquet"),
    split="all"
)

dataset_eval = load_dataset(
    path="parquet",
    data_files=os.path.join(path_data, "tatsu-lab/alpaca/dataset_eval_2.parquet"),
    split="all"
)

# ----------------------------------------------------------------------------------------------------------------
# LLM
# https://huggingface.co/Salesforce/xgen-7b-8k-base
# https://huggingface.co/facebook/opt-350m
# https://huggingface.co/THUDM/chatglm3-6b
# checkpoint = "facebook/opt-350m"
# checkpoint = "gpt2"
checkpoint = "chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
)

# tokenizer.pad_token  # '<unk>'
# tokenizer.eos_token  # '</s>'
# tokenizer.pad_token = tokenizer.eos_token  # 半精度训练时需要
# tokenizer.padding_side = "right"  # llama2
# tokenizer.build_chat_input(query, history=[], role="user")  # chatGLM3
# tokenizer.decode(token_ids=ids)
len(tokenizer.get_vocab())  # 64796

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
    quantization_config=config_bnb
)
'''
model: 6B (fp32)
weight: 6G * 4 = 24
grads: 6G * 4 = 24
optim: 6G * 4 * 2 = 48
'''

# for param in model_base.parameters():
#     param.requires_grad_(False)

# model_base.is_parallelizable = True
# model_base.model_parallel = True
model_base.config.use_cache = False

for i, (name, parm) in enumerate(model_base.named_parameters()):
    print(f"{i}  name: {name};  shape: {parm.shape};  dtype: {parm.dtype};  device: {parm.device}")
'''
0  name: model.decoder.embed_tokens.weight;  shape: torch.Size([50272, 512]);  dtype: torch.bfloat16;  device: cuda:0
1  name: model.decoder.embed_positions.weight;  shape: torch.Size([2050, 1024]);  dtype: torch.bfloat16;  device: cuda:0
2  name: model.decoder.project_out.weight;  shape: torch.Size([512, 1024]);  dtype: torch.bfloat16;  device: cuda:0

385  name: model.decoder.layers.23.fc2.bias;  shape: torch.Size([1024]);  dtype: torch.bfloat16;  device: cuda:0
386  name: model.decoder.layers.23.final_layer_norm.weight;  shape: torch.Size([1024]);  dtype: torch.bfloat16;  device: cuda:0
387  name: model.decoder.layers.23.final_layer_norm.bias;  shape: torch.Size([1024]);  dtype: torch.bfloat16;  device: cuda:0
'''

print(model_base)
print(model_base.dtype)

# check embedding_size
embedding_size = model_base.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model_base.resize_token_embeddings(len(tokenizer))

# ----------------------------------------------------------------------------------------------------------------
# model config
config_model = {
    "rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "use_rslora": True,
    "epochs": 10,
    "batch_size": 4,
    "gradient_steps": 1,
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
    task_type=TaskType.CAUSAL_LM
)

# model_base = prepare_model_for_int8_training(model_base)
# windows 环境：https://github.com/jllllll/bitsandbytes-windows-webui/tree/wheels
model_lora = get_peft_model(model=model_base, peft_config=config_lora)
model_lora.enable_input_require_grads()  # if TrainingArguments(gradient_checkpointing=True)
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

print(
    f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")

# ----------------------------------------------------------------------------------------------------------------
# SFT
# train
args_train = TrainingArguments(
    output_dir=os.path.join(path_model, "model_sft"),  # 输出目录
    num_train_epochs=3,  # 训练轮数
    per_device_train_batch_size=4,  # 训练批次大小
    per_device_eval_batch_size=4,  # 验证批次大小
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    learning_rate=0.001,  # 0.00005
    # weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    save_strategy="steps",
    evaluation_strategy="steps",
    log_level="info",
    logging_strategy="steps",
    logging_steps=10,  # 500
    eval_steps=10,
    logging_dir=path_log,
    report_to="all",
    load_best_model_at_end=False,
    remove_unused_columns=False,
    # push_to_hub=False
)

args_train = TrainingArguments(
    output_dir=os.path.join(path_model, "model_sft"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,  # save mem but waste time
    gradient_checkpointing=True,    # save mem but waste time
    optim="adafactor",              # save mem but waste time, paged_adamw_32bit
    learning_rate=0.001,
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=3,
    metric_for_best_model="f1",
    load_best_model_at_end=True
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
writer = SummaryWriter()

trainer = SFTTrainer(
    model=model_lora,
    tokenizer=tokenizer,
    args=args_train,
    peft_config=config_lora,
    data_collator=collate_fn,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    dataset_text_field="text",  # 用于指示数据集中哪个字段包含作为模型输入的文本数据
    packing=True,
    max_seq_length=512,
    callbacks=[TensorBoardCallback(writer)]
    # compute_metrics=compute_metrics,
)

'''
## compute_metrics
from sklearn.metrics import accuracy_score, f1_score  
def compute_metrics(pred):  
    labels = pred.label_ids  
    preds = pred.predictions.argmax(-1)  
    accuracy = accuracy_score(labels, preds)  
    f1 = f1_score(labels, preds, average='macro')  # 或使用 'micro', 'weighted' 等  
    return {'accuracy': accuracy, 'f1': f1}
'''

# model_lora.config.use_cache = False
output_train = trainer.train()
metrics = output_train.metrics
'''
TrainOutput(global_step=18, training_loss=1.8800998263888888, 
            metrics={'train_runtime': 2613.6248, 'train_samples_per_second': 0.028, 
                     'train_steps_per_second': 0.007, 'total_flos': 1322178922414080.0, 
                     'train_loss': 1.8800998263888888, 'epoch': 3.0})
'''
output_eval = trainer.evaluate(dataset_train)
output_eval = trainer.evaluate(dataset_eval)
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
