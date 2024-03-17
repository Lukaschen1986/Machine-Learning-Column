# -*- coding: utf-8 -*-
'''
https://huggingface.co/course/zh-CN/chapter2/3?fw=pt
'''
import os
import random
import numpy as np
import torch as th
from torch import nn
from torch.utils.data import random_split
from transformers import (pipeline, 
                          AutoTokenizer, BertTokenizer,
                          AutoModel, BertModel, BertConfig,
                          AutoModelForSequenceClassification, 
                          BertForSequenceClassification,
                          AdamW, AutoConfig)
from transformers import (DataCollatorWithPadding, DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq, default_data_collator)
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
# 1-从默认配置创建模型会使用随机值对其进行初始化
checkpoint = "bert-base-chinese"

# config = AutoConfig.from_pretrained(
#     pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
#     cache_dir=path_model,
#     force_download=False,
#     local_files_only=False
#     )  # 1
# config = BertConfig()  # 2
# pretrained = BertModel(config)

# 2-加载已经训练过的Transformers模型
pretrained = BertModel.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
    )
print(pretrained)
print(pretrained.config)
print(pretrained.config.output_attentions)  # False
print(pretrained.config.output_hidden_states)  # False

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
# 简单案例：不带Model Head的模型调用
tokenizer = AutoTokenizer.from_pretrained("rbt3")
sen = "弱小的我也有大梦想！"
inputs = tokenizer(sen, return_tensors="pt")

model = AutoModel.from_pretrained("rbt3", output_attentions=True)  # 强制加上 output_attentions=True 输出 attentions
output = model(**inputs)
print(output.last_hidden_state.shape())  # torch.Size([1, 12, 768])

# 简单案例：带Model Head的模型调用
model = AutoModelForSequenceClassification.from_pretrained("rbt3", num_labels=10)
output = model(**inputs)
print(output.logits.shape())  # torch.Size([1, 10])

# ----------------------------------------------------------------------------------------------------------------
# 中文分类
# 数据集类
class Dataset(th.utils.data.Dataset):
    def __init__(self):
        # super(Dataset, self).__init__()
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

# 划分数据集
dataset = Dataset()
trainset, validset = random_split(dataset, lengths=[0.9, 0.1])
# trainset, validset = dataset.train_test_split(test_size=0.1, stratify_by_column="label") 

# 整理函数
def collate_fn(dataset):
    texts = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]
    max_length = max(len(x) for x in texts) + 2
    
    #编码
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                          truncation=True,
                                          padding="max_length",
                                          max_length=max_length,
                                          return_tensors="pt",
                                          return_length=True)

    labels = th.LongTensor(labels)  # torch.int64
    # labels = th.Tensor(labels, dtype=th.long)  # torch.int64
    return inputs, labels
# collate_fn = DataCollatorWithPadding(tokenizer)

# 数据迭代器
loader = th.utils.data.DataLoader(dataset=dataset,
                                  batch_size=16,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)

# 下游模型
config = {"pretrained": pretrained}

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

# 训练
model = Model(config).to(device)
opti = optim.AdamW(params=model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
# opti = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
# objt = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="mean")
objt = nn.CrossEntropyLoss(reduction="mean")
epochs = 200

for epoch in range(epochs):
    # train
    loss_tmp = 0
    model.train()
    for (i, (inputs, labels)) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        output_mlp = model(inputs)
        loss = objt(output_mlp, labels)
        loss_tmp += loss.item()
        
        opti.zero_grad(set_to_none=True)
        loss.backward()
        opti.step()
    
    loss_train = loss_tmp / (i+1)
    print(f"epoch {epoch}  loss_train {loss_train:.4f}")

# 推理
max_length = len(texts)
inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                     truncation=True,
                                     padding="max_length",
                                     max_length=max_length,
                                     return_tensors="pt",
                                     return_length=True)

model.eval()
with th.no_grad():
    out_mlp = model(inputs)
    y_hat = th.softmax(out_mlp, dim=1)
    y_pred = th.argmax(y_hat, dim=1)


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
def collate_fn(data_set):
    sents = [x[0:2] for x in data_set]  # 适用于定义数据集
    labels = [x[2] for x in data_set]  # 适用于定义数据集
    # sents = [(dct.get("sent1"), dct.get("sent2")) for dct in data_set]  # 适用于 Dataset.from_pandas()
    # labels = [dct.get("label") for dct in data_set]  # 适用于 Dataset.from_pandas()
    max_length = max(len(x[0]) + len(x[1]) for x in sents) + 3

    #编码
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                         truncation=True,
                                         padding="max_length",
                                         max_length=max_length,
                                         add_special_tokens=True,
                                         return_token_type_ids=True,
                                         return_attention_mask=True,
                                         return_special_tokens_mask=True,
                                         return_tensors="pt",
                                         return_length=True)
    labels = th.LongTensor(labels)
    return inputs, labels


# 数据迭代器
data_set = Dataset()
loader = th.utils.data.DataLoader(dataset=data_set,
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
    

    
