# -*- coding: utf-8 -*-
"""
https://github.com/huggingface/trl
https://huggingface.co/docs/trl/reward_trainer
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
from trl import (RewardConfig, RewardTrainer)


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

# ----------------------------------------------------------------------------------------------------------------
# path
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")


'''
# imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

...

# load trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

# train
trainer.train()
'''