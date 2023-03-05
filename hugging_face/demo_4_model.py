# -*- coding: utf-8 -*-
'''
https://huggingface.co/course/zh-CN/chapter2/3?fw=pt
'''
import os
import random
import numpy as np
import torch as th
from torch import nn
from transformers import (pipeline, 
                          AutoTokenizer, BertTokenizer,
                          AutoModel, BertModel, BertConfig,
                          AutoModelForSequenceClassification, 
                          AdamW)
from transformers import (DataCollatorWithPadding, default_data_collator)
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
# 加载预训练模型
# 从默认配置创建模型会使用随机值对其进行初始化
config = BertConfig()
pretrained = BertModel(config)

# 加载已经训练过的Transformers模型
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
checkpoint = "bert-base-chinese"
# checkpoint = "hfl/rbt6"
# checkpoint = "hfl/chinese-bert-wwm-ext"
# checkpoint = "nghuyong/ernie-3.0-base-zh"
# checkpoint = "nghuyong/ernie-1.0-base-zh"

pretrained = BertModel.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
    )

for param in pretrained.parameters():
    param.requires_grad_(False)
    
'''
权重已下载并缓存在缓存文件夹中（因此将来对from_pretrained()方法的调用将不会重新下载它们）
默认为 ~/.cache/huggingface/transformers . 您可以通过设置 HF_HOME 环境变量来自定义缓存文件夹。
'''

pretrained.save_pretrained(save_directory=os.path.join(path_model, "my_model_dir"))
'''
如果你看一下 config.json 文件，您将识别构建模型体系结构所需的属性。
该文件还包含一些元数据，例如检查点的来源以及上次保存检查点时使用的🤗 Transformers版本。

这个 pytorch_model.bin 文件就是众所周知的state dictionary; 它包含模型的所有权重。
这两个文件齐头并进；配置是了解模型体系结构所必需的，而模型权重是模型的参数。
'''

# ----------------------------------------------------------------------------------------------------------------
# 中文分类
# 数据集类
class Dataset(th.utils.data.Dataset):
    def __init__(self):
        self.dataset = load_dataset(
            path="csv",
            data_files=os.path.join(path_data, "text_cls.csv"),
            split="all"
            )
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]["text"]
        label = self.dataset[i]["label"]
        return text, label 

# 整理函数
def collate_fn(dataset):
    texts = [i[0] for i in dataset]
    labels = [i[1] for i in dataset]
    
    #编码
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                         truncation=True,
                                         padding="max_length",
                                         max_length=500,
                                         return_tensors="pt",
                                         return_length=True)

    labels = th.LongTensor(labels)
    return inputs, labels

# 数据迭代器
loader = th.utils.data.DataLoader(dataset=Dataset(),
                                  batch_size=16,
                                  collate_fn=collate_fn,  # 简单: DataCollatorWithPadding(tokenizer)
                                  shuffle=True,
                                  drop_last=True)

# 下游模型
class Model(th.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pretrained = config.get("pretrained")
        self.mlp = th.nn.Linear(768, 2)  # 此处可设计为多层mlp，用nn.Sequential包裹

    def forward(self, inputs):
        tokens = inputs["input_ids"]
        segments = inputs["token_type_ids"]
        valid_lens = inputs["attention_mask"]
        
        output_bert = self.pretrained(
            input_ids=tokens,
            token_type_ids=segments,
            attention_mask=valid_lens
            ).last_hidden_state

        out_mlp = self.mlp(output_bert[:, 0, :])
        return out_mlp

# ----------------------------------------------------------------------------------------------------------------
# 完形填空
# 数据集类
class Dataset(th.utils.data.Dataset):
    def __init__(self):
        dataset = load_dataset(
            path="csv",
            data_files=os.path.join(path_data, "text_fill.csv"),
            split="all"
            )

        def func(dataset):
            return len(dataset["text"]) > 30
        self.dataset = dataset.filter(func)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]["text"]
        return text

# 整理函数
def collate_fn(dataset):
    #编码
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=dataset,
                                         truncation=True,
                                         padding="max_length",
                                         max_length=30,
                                         return_tensors="pt",
                                         return_length=True)

    tokens = inputs["input_ids"]
    segments = inputs["token_type_ids"]
    valid_lens = inputs["attention_mask"]

    # 把第15个词固定替换为mask
    labels = tokens[:, 15].reshape(-1).clone()
    tokens[:, 15] = tokenizer.mask_token_id
    labels = th.LongTensor(labels)
    return tokens, segments, valid_lens, labels

# 数据迭代器
loader = th.utils.data.DataLoader(dataset=Dataset(),
                                  batch_size=16,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)

# 下游模型
class Model(th.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pretrained = config.get("pretrained")
        self.mlp = th.nn.Linear(768, tokenizer.vocab_size, bias=False)
        bias = th.nn.Parameter(th.zeros(tokenizer.vocab_size))  # 可选
        self.mlp.bias = bias  # 可选

    def forward(self, tokens, segments, valid_lens):
        output_bert = self.pretrained(
            input_ids=tokens,
            token_type_ids=segments,
            attention_mask=valid_lens
            ).last_hidden_state

        out_mlp = self.mlp(output_bert[:, 15, :])
        return out_mlp

# ----------------------------------------------------------------------------------------------------------------
# 中文句子关系推断
# 数据集类
class Dataset(th.utils.data.Dataset):
    def __init__(self):
        dataset = load_dataset(
            path="csv",
            data_files=os.path.join(path_data, "text_inference.csv"),
            split="all"
            )

        def func(data):
            return len(data["text"]) > 40
        self.dataset = dataset.filter(func)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]["text"]
        sent1 = text[0:20]
        
        # 有一半的概率把后半句替换为一句无关的话
        if random.random() < 0.5:
            j = random.choice(range(self.dataset))
            sent2 = self.dataset[j]["text"][20:40]
            label = 1
        else:
            sent2 = text[20:40]
            label = 0

        return sent1, sent2, label


# 整理函数
def collate_fn(dataset):
    sents = [x[0:2] for x in dataset]
    labels = [x[2] for x in dataset]

    #编码
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                         truncation=True,
                                         padding="max_length",
                                         max_length=45,
                                         return_tensors="pt",
                                         return_length=True,
                                         add_special_tokens=True)
    
    labels = th.LongTensor(labels)
    return inputs, labels


# 数据迭代器
loader = th.utils.data.DataLoader(dataset=Dataset(),
                                  batch_size=16,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)

# 下游模型
class Model(th.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pretrained = config.get("pretrained")
        self.mlp = th.nn.Linear(768, 2)

    def forward(self, inputs):
        output_bert = self.pretrained(
            input_ids=tokens,
            token_type_ids=segments,
            attention_mask=valid_lens
            ).last_hidden_state

        out_mlp = self.mlp(output_bert[:, 0, :])
        return out_mlp
    

    
