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

# ----------------------------------------------------------------------------------------------------------------
# load dataset
# https://huggingface.co/datasets/Anthropic/hh-rlhf
dataset_train = load_dataset(
    path="arrow",
    data_files=os.path.join(path_data, "Anthropic/hh-rlhf/hh-rlhf-train.arrow"),
    split="train"
    )
print(dataset_train)
'''
Dataset({
    features: ['chosen', 'rejected'],
    num_rows: 160800
})
'''

dataset_eval = load_dataset(
    path="arrow",
    data_files=os.path.join(path_data, "Anthropic/hh-rlhf/hh-rlhf-test.arrow"),
    split="all"
    )
print(dataset_eval)

print(dataset_train[0])
'''
{
 'chosen': (
    "\n\nHuman: What are some cuss words in english?"
    "\n\nAssistant: Here’s an incomplete list."
    "\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, "
    "shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, "
    "sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, "
    "cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, "
    "queer, scumbag, bitch,"
    "\n\nHuman: What's your favorite one?"
    "\n\nAssistant: I haven't even thought about it."
    ), 
 'rejected': (
     "\n\nHuman: What are some cuss words in english?"
     "\n\nAssistant: Here’s an incomplete list."
     "\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, "
     "shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, "
     "sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, "
     "cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, "
     "queer, scumbag, bitch,"
     "\n\nHuman: What's your favorite one?"
     "\n\nAssistant: Ass.")
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
len(tokenizer.get_vocab())  # 50257

config_bnb = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=False,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=th.bfloat16
    )

model_reward = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=th.bfloat16,
    # quantization_config=config_bnb,
    num_labels=1  # reward need
    )

# for param in model_reward.parameters():
#     param.requires_grad_(False)

for i, (name, parm) in enumerate(model_reward.named_parameters()):
    print(f"{i}  name: {name};  shape: {parm.shape};  dtype: {parm.dtype};  device: {parm.device}")

print(model_reward)
print(model_reward.score)

# ----------------------------------------------------------------------------------------------------------------
# transform datasets
max_length = 1024

def preprocess_function(sample):
    new_sample = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    
    for (chosen, rejected) in zip(sample["chosen"], sample["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)
        new_sample["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_sample["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_sample["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_sample["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_sample


datapair_train = dataset_train.map(
    function=preprocess_function,
    batched=True,
    num_proc=1  # 4
    )

datapair_train = datapair_train.filter(
    lambda x: len(x["input_ids_chosen"]) <= max_length
    and len(x["input_ids_rejected"]) <= max_length
)

datapair_eval = dataset_eval.map(
    function=preprocess_function,
    batched=True,
    num_proc=1  # 4
    )

datapair_eval = datapair_eval.filter(
    lambda x: len(x["input_ids_chosen"]) <= max_length
    and len(x["input_ids_rejected"]) <= max_length
)

# ----------------------------------------------------------------------------------------------------------------
# LoRA
config_lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
    )

# model_reward = prepare_model_for_int8_training(model_reward)
model_lora = get_peft_model(model=model_reward, peft_config=config_lora)
print(model_lora)

model_lora.print_trainable_parameters()
'''
trainable params: 295,680 || all params: 124,736,256 || trainable%: 0.23704415178214103
'''

# ----------------------------------------------------------------------------------------------------------------
# Reward
# train
args_reward = RewardConfig(
    output_dir=os.path.join(path_model, "model_reward"),
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
    gradient_checkpointing=True
    )

trainer = RewardTrainer(
        model=model_lora,
        tokenizer=tokenizer,
        args=args_reward,
        peft_config=config_lora,
        train_dataset=datapair_train,
        eval_dataset=datapair_eval,
        max_length=max_length,
        # compute_metrics=
    )

trainer.train()
trainer.evaluate()

trainer.save_model(os.path.join(path_model, "model_reward.bin"))


