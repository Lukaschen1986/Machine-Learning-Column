# -*- coding: utf-8 -*-
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

# ----------------------------------------------------------------------------------------------------------------
# LLM
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
tokenizer.pad_token = tokenizer.eos_token  # 半精度训练时需要
# tokenizer.padding_side = "right"  # llama2

model_base = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=th.bfloat16
)

# ----------------------------------------------------------------------------------------------------------------
# prompt
schema = """
{
 "infoList": [
    {
     "ssid": "string",
     "securityProtocol": "string",
     "bandwidth": "string"
    }
    ]
}
"""
schema = schema.replace("\n", "").replace(" ", "")

content_sys = (
    "You are a helpful assistant that answers in JSON. "
    f"Here's the json schema you must adhere to: \n<schema>\n{schema}\n</schema>\n"
    )
print(content_sys)

content_usr = (
    "I'm currently configuring a wireless access point for our office network and I "
    "need to generate a JSON object that accurately represents its settings. "
    "The access point's SSID should be 'OfficeNetSecure', it uses WPA2-Enterprise "
    "as its security protocol, and it's capable of a bandwidth of up to 1300 Mbps "
    "on the 5 GHz band. This JSON object will be used to document our network "
    "configurations and to automate the setup process for additional access "
    "points in the future. Please provide a JSON object that includes these details."
    )
len(content_usr.split(" "))

messages = [
    {"role": "system", "content": content_sys},
    {"role": "user", "content": content_usr}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)

# ----------------------------------------------------------------------------------------------------------------
# inference
max_new_tokens = 128  # 取训练样本答案的最长值
top_p = 0.9
temperature = 1.0  # 0.5，0.35，0.1，0.01
repetition_penalty = 1.5

model_inputs = tokenizer([text], return_tensors="pt").to(device)

t0 = pd.Timestamp.now()
model_base.eval()
with th.inference_mode():
    generated_ids = model_base.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        # top_p=top_p,
        # temperature=temperature,
        # repetition_penalty=repetition_penalty
    )
t1 = pd.Timestamp.now()
print(t1 - t0)

generated_ids = [O[len(I): ] for (I, O) in zip(model_inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

response.split("```")[1].replace("\n", "").replace(" ", "")  # sft时答案加上 \n<schema>\n{response}\n</schema>\n
