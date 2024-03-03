# -*- coding: utf-8 -*-
"""
https://github.com/huggingface/trl
https://huggingface.co/docs/trl/ppo_trainer
pip install -U datasets transformers accelerate peft trl --user
pip install -U bitsandbytes --user
https://github.com/jllllll/bitsandbytes-windows-webui/tree/wheels
https://zhuanlan.zhihu.com/p/635757674
https://zhuanlan.zhihu.com/p/631419889
"""
import os
import sys
import numpy as np
from tqdm import tqdm
import torch as th
from datasets import (load_dataset, load_from_disk, Dataset)
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          BitsAndBytesConfig, TrainingArguments, DataCollatorWithPadding, 
                          DataCollatorForLanguageModeling, pipeline)
from peft import (LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_int8_training)
from trl import (PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model)


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------
# path
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# load dataset
# https://huggingface.co/datasets/HuggingFaceH4/cherry_picked_prompts
dataset_train = load_dataset(
    path="parquet",
    data_files=os.path.join(path_data, "cherry_picked_prompts/train-00000-of-00001-644a4ce71cfb940a.parquet"),
    split="train"
    )
dataset_train = dataset_train.rename_column("prompt", "query")
dataset_train = dataset_train.remove_columns(["meta", "completion"])

print(dataset_train)
'''
Dataset({
    features: ['query'],
    num_rows: 16
})
'''

print(dataset_train[0:3])
'''
{
 'query': [
     'Explain the moon landing to a 6 year old in a few sentences.', 
     'Q: Who was president of the United States in 1955? A: Dwight D. Eisenhower was president of the United States in 1955. Q: How does a telescope work? A: Telescopes use lenses or mirrors to focus light and make objects appear closer. Q: Why do birds migrate south for the winter? A:', 
     'Why aren’t birds real?'
     ]
 }
'''

# ----------------------------------------------------------------------------------------------------------------
# LLM
# https://huggingface.co/openai-community/gpt2
checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
    )
tokenizer.pad_token = tokenizer.eos_token

config_bnb = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=False,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=th.bfloat16
    )

model_act = AutoModelForCausalLMWithValueHead.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=th.bfloat16,
    # quantization_config=config_bnb,
    )
# model_act = AutoModelForCausalLMWithValueHead(model_sft)
print(model_act.v_head)
'''
ValueHead(
  (dropout): Dropout(p=0.1, inplace=False)
  (summary): Linear(in_features=512, out_features=1, bias=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
)
'''
model_ref = create_reference_model(model_act)

model_reward = pipeline("text-classification", model="lvwerra/distilbert-imdb")
# model_reward = th.load(os.path.join(path_model, "model_reward.bin"))

# ----------------------------------------------------------------------------------------------------------------
# transform datasets
# def tokenize(sample):
#     sample["input_ids"] = tokenizer.encode(sample["query"])
#     return sample

# dataset_train = dataset_train.map(function=tokenize, batched=False)

# ----------------------------------------------------------------------------------------------------------------
# PPO
# train
args_ppo = PPOConfig(batch_size=1)

trainer = PPOTrainer(
    model=model_act,
    ref_model=model_ref,
    tokenizer=tokenizer,
    dataset=dataset_train,
    config=args_ppo
    )

generation_kwargs = {
    "min_length": -1,  # don't ignore the EOS token (see above)
    "top_k": 0.0,  # no top-k sampling
    "top_p": 1.0,  # no nucleus sampling
    "do_sample": True,  # yes, we want to sample
    "pad_token_id": tokenizer.eos_token_id,  # most decoder models don't have a padding token - use EOS token instead
    "max_new_tokens": 64,  # specify how many tokens you want to generate at most
}

epochs = trainer.config.ppo_epochs
loader = trainer.dataloader

#### 单样本版本
for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(loader):
        query_txt = batch["query"][0]
        query_pt = tokenizer.encode(query_txt, return_tensors="pt").to(device)
        
        #### Get response from SFTModel
        response_pt = trainer.generate(query_pt.squeeze(), return_prompt=False, **generation_kwargs)
        batch["response"] = [tokenizer.decode(response_pt.squeeze())]
        
        #### Compute reward score
        texts = [q + r for (q, r) in zip(batch["query"], batch["response"])]
        pipe_outputs = model_reward(texts)
        reward = [th.tensor(dct["score"]) for dct in pipe_outputs]
        
        #### Run PPO step
        stats = trainer.step([query_pt[0]], [response_pt[0]], reward)
        trainer.log_stats(stats, batch, reward)

#### 多样本版本
# for epoch in tqdm(range(epochs), "epoch: "):
#     for batch in tqdm(loader): 
#         query_tensors = batch["input_ids"]
        
#         #### Get response from SFTModel
#         response_tensors = trainer.generate(query_tensors, **generation_kwargs)
#         batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
#         #### Compute reward score
#         texts = [q + r for q, r in zip(batch["query"], batch["response"])]
#         pipe_outputs = model_reward(texts)
#         rewards = [th.tensor(output[1]["score"]) for output in pipe_outputs]
    
#         #### Run PPO step
#         stats = trainer.step(query_tensors, response_tensors, rewards)
#         trainer.log_stats(stats, batch, rewards)


#### Save model
trainer.save_model(os.path.join(path_model, "model_ppo.bin"))

'''
# imports
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch

# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
model_ref = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained('gpt2')

# initialize trainer
ppo_config = PPOConfig(batch_size=1)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(device)

# get model response
response_tensor = respond_to_batch(model_act, query_tensor)

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
stats = trainer.step([query_tensor[0]], [response_tensor[0]], rewards)

[query_tensor[0]]
[tensor([1212, 3329,  314, 1816,  284,  262,  220], device='cuda:0')]

[response_tensor[0]]
[tensor([  933, 12754,  1295,   503,   329,  1573,   286,   428, 45142,   560,
           284,   766,   644,   262, 23788,   547,  1804,   612,    13,   314],
        device='cuda:0')]
'''