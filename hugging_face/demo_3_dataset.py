# -*- coding: utf-8 -*-
import os
import numpy as np
import torch as th
from torch import nn
from transformers import (pipeline, 
                          AutoTokenizer, BertTokenizer,
                          AutoModel, BertModel, BertConfig,
                          AutoModelForSequenceClassification, 
                          AdamW)
from datasets import (load_dataset, load_from_disk, Dataset)
from torchcrf import CRF
import torch.optim as optim

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 加载数据
dataset = load_from_disk(dataset_path=os.path.join(path_data, "peoples_daily_ner"))
dataset = load_dataset("madao33/new-title-chinese", split="train[10:100]")
dataset = load_dataset("madao33/new-title-chinese", split="train[:50%]")
dataset = load_dataset("madao33/new-title-chinese", split=["train[:50%]", "validation[:10%]"])

dataset = load_dataset(
    path="csv",  # (JSON, CSV, Parquet, text, etc.)
    data_files=os.path.join(path_data, "peoples_daily_ner.csv"),
    split="all"
    )
# dataset = Dataset.from_csv(path_or_paths=os.path.join(path_data, "peoples_daily_ner.csv"), split="all")
dataset = load_dataset(path="csv", data_dir=path_data, split="all")

features = [...]
data_set = Dataset.from_pandas(df, features)\
    .train_test_split(test_size=0.15, shuffle=True, seed=0)
train_set = data_set.get("train")

# ----------------------------------------------------------------------------------------------------------------
# 数据集基本操作
print(dataset["train"][0])
'''
{'title': '...',
 'content': '...'}
'''
print(dataset["train"][0:2])
'''
{'title': ['...', '...'],
 'content': ['...', '...']}
'''
print(dataset["train"]["title"][0:2])
'''
['...', '...']
'''

dataset["train"].column_names  # ['title', 'content']
dataset["train"].features
'''
{'title': Value(dtype='string', id=None),
 'content': Value(dtype='string', id=None)}
'''

dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.train_test_split(test_size=0.1, stratify_by_column="label", shuffle=True, seed=0) 

dataset["train"].select([0, 1])  # 选取0,1行
filter_dataset = dataset["train"].filter(lambda example: "中国" in example["title"])

def add_prefix(example):
    example["title"] = 'Prefix: ' + example["title"]
    return example
prefix_dataset = dataset.map(function=add_prefix)

def preprocess_function(example, tokenizer):
    model_inputs = tokenizer(example["content"], max_length=512, truncation=True)
    labels = tokenizer(example["title"], max_length=32, truncation=True)
    # label就是title编码的结果
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
# processed_dataset = dataset.map(preprocess_function, num_proc=4)
processed_dataset = dataset.map(preprocess_function, batched=True)
'''
DatasetDict({
    train: Dataset({
        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 5850
    })
    validation: Dataset({
        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 1679
    })
})
'''
processed_dataset = dataset.map(preprocess_function, batched=True, 
                                remove_columns=dataset["train"].column_names)
'''
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 5850
    })
    validation: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 1679
    })
})
'''
processed_dataset.save_to_disk("./processed_data")
processed_dataset = load_from_disk("./processed_data")

# ----------------------------------------------------------------------------------------------------------------
# 自定义数据集
class Dataset(th.utils.data.Dataset):
    def __init__(self):
        self.dataset = load_dataset(
            path="csv",
            data_files=os.path.join(path_data, "ner.csv"),
            split="all"
            )
        
        # 自定义filter函数（可选）
        def func(data):
            return len(data["tokens"]) <= 512 - 2
        self.dataset = dataset.filter(func)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        tokens = self.dataset[i]["tokens"]
        labels = self.dataset[i]["ner_tags"]
        return tokens, labels 


class Dataset(th.utils.data.Dataset):
    def __init__(self, split="train"):
        dataset = load_dataset(
            path="csv",
            data_files=os.path.join(path_data, "ner.csv"),
            split="all"
            )
        self.dataset = dataset.train_test_split(test_size=0.1)
        self.split = split
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        tokens = self.dataset[self.split][i]["tokens"]
        labels = self.dataset[self.split][i]["ner_tags"]
        return tokens, labels 
    

class Dataset(th.utils.data.Dataset):
    def __init__(self, split=None):
        if split == "train":
            self.dataset = load_dataset(data_files=os.path.join(path_data, "df_train.parquet"),
                                        path="parquet", split="all")
        elif split == "test":
            self.dataset = load_dataset(data_files=os.path.join(path_data, "df_test.parquet"),
                                        path="parquet", split="all")
        elif (not split) or (split == "all"):
            self.dataset = load_dataset(data_files=os.path.join(path_data, "df.parquet"),
                                        path="parquet", split="all")
        else:
            raise ValueError
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        sent1 = self.dataset[i]["a"]
        sent2 = self.dataset[i]["b"]
        label = self.dataset[i]["label"]
        return sent1, sent2, label

df.to_parquet(os.path.join(path_data, "df.parquet"), index=False)
data_set = Dataset(split="all")