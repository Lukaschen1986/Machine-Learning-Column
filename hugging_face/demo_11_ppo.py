# -*- coding: utf-8 -*-
"""
https://github.com/huggingface/trl
https://huggingface.co/docs/trl/ppo_trainer
pip install -U datasets transformers accelerate peft trl --user
pip install -U bitsandbytes --user
https://github.com/jllllll/bitsandbytes-windows-webui/tree/wheels
"""
import os
import sys
import numpy as np
import torch as th
from datasets import (load_dataset, load_from_disk, Dataset)
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          BitsAndBytesConfig, TrainingArguments, DataCollatorWithPadding, 
                          DataCollatorForLanguageModeling)
from peft import (LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_int8_training)
from trl import (PPOConfig, PPOTrainer)


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------
# path
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

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
ppo_config = PPOConfig(
    batch_size=1,
)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(model, query_tensor)

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
'''