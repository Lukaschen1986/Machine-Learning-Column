# -*- coding: utf-8 -*-
'''
https://huggingface.co/course/zh-CN/chapter2/2?fw=pt
'''
import os
import numpy as np
import torch as th
from torch import nn
from transformers import (pipeline, 
                          AutoTokenizer, BertTokenizer,
                          AutoModel, BertModel, BertConfig,
                          AutoModelForSequenceClassification, 
                          AdamW)
from datasets import (load_dataset, load_from_disk)
from torchcrf import CRF
import torch.optim as optim

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 加载分词器
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# checkpoint = "bert-base-chinese"
# checkpoint = "hfl/rbt6"
# checkpoint = "hfl/chinese-bert-wwm-ext"
# checkpoint = "nghuyong/ernie-3.0-base-zh"
# checkpoint = "nghuyong/ernie-1.0-base-zh"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=True
)

# ----------------------------------------------------------------------------------------------------------------
# 简单编码
raw_inputs = "Using a Transformer network is simple"

tokens = tokenizer.tokenize(text=raw_inputs)
'''
['using', 'a', 'transform', '##er', 'network', 'is', 'simple']
'''

ids = tokenizer.convert_tokens_to_ids(tokens)
'''
[2478, 1037, 10938, 2121, 2897, 2003, 3722]
'''

tokens = tokenizer.convert_ids_to_tokens(ids)
'''
['using', 'a', 'transform', '##er', 'network', 'is', 'simple']
'''

raw_inputs = tokenizer.decode(token_ids=ids)
'''
'using a transformer network is simple'

请注意， decode 方法不仅将索引转换回标记(token)，还将属于相同单词的标记(token)组合在一起以生成可读的句子。
当我们使用预测新文本的模型（根据提示生成的文本，或序列到序列问题（如翻译或摘要））时，这种行为将非常有用。
'''

# ----------------------------------------------------------------------------------------------------------------
# 稍复杂编码
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

inputs = tokenizer(
    text=raw_inputs, 
    padding=True,  # False==do_not_pad, True==longest, max_length
    truncation=True, 
    return_tensors="pt"
    )

tokenizer.decode(inputs["input_ids"][1])
'''
"[CLS] i hate this so much! [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"
'''

# ----------------------------------------------------------------------------------------------------------------
# 复杂编码
inputs = tokenizer.encode(
    text=raw_inputs[0],
    text_pair=raw_inputs[1],
    max_length=30,
    padding="max_length",
    truncation=True,
    add_special_tokens=True,
    return_tensors=None
    )

inputs = tokenizer.encode_plus(
    text=raw_inputs[0],
    text_pair=raw_inputs[1],  # None
    max_length=30,
    padding="max_length",
    truncation=True,
    add_special_tokens=True,
    return_tensors=None,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True
    )

inputs = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[raw_inputs[0], raw_inputs[1]],  # [(sents[0], sents[1]), (sents[2], sents[3])]
    max_length=15,
    padding="max_length",
    add_special_tokens=True,
    truncation=True,
    return_tensors=None,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True,
    is_split_into_words=True
    )

dct = tokenizer.get_vocab()
tokenizer.decode(inputs["input_ids"][0])

# ----------------------------------------------------------------------------------------------------------------
# 加载预训练模型
pretrained = AutoModel.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
    )

outputs = pretrained(**inputs)
print(outputs.last_hidden_state.shape)  # torch.Size([2, 16, 768])
# print(outputs[0].shape)

pretrained = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
    )

outputs = pretrained(**inputs)
print(outputs.logits.shape)  # torch.Size([2, 2])
# print(outputs[0].shape)


