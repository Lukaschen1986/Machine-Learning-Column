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

dataset = load_dataset(
    path="csv",
    data_files=os.path.join(path_data, "peoples_daily_ner.csv"),
    split="all"
    )

dataset = load_dataset(
    path="json",
    data_files=os.path.join(path_data, "peoples_daily_ner.json"),
    split="all"
    )

features = [...]
data_set = Dataset.from_pandas(df, features).train_test_split(test_size=0.15, shuffle=True, seed=0)
train_set = data_set.get("train")

# ----------------------------------------------------------------------------------------------------------------
# 定义数据集
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