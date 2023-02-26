# -*- coding: utf-8 -*-
import os
import numpy as np
import torch as th
from torch import nn
from transformers import (BertTokenizer, AutoTokenizer, BertModel, AdamW)
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
# checkpoint = "bert-base-chinese"
checkpoint = "hfl/rbt6"
# checkpoint = "hfl/chinese-bert-wwm-ext"
# checkpoint = "nghuyong/ernie-3.0-base-zh"
# checkpoint = "nghuyong/ernie-1.0-base-zh"

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=True
)

# ----------------------------------------------------------------------------------------------------------------
# 加载预训练模型
pretrained = BertModel.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=True
    )

for param in pretrained.parameters():
    param.requires_grad_(False)

batch_size = 50

config = {
    "pretrained": pretrained,
    "embedding_dim": 768,
    "hidden_dim": 256,
    "tagset_size": 8,
    "dropout": 0.2,
    "batch_size": batch_size,
    "device": device
    }

# ----------------------------------------------------------------------------------------------------------------
sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷,真有点郁闷.',
    '机器背面似乎被撕了张什么标签，残胶还在。',
]

# 简单解码，一对句子
inputs = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],
    max_length=30,
    padding="max_length",
    truncation=True,
    add_special_tokens=True,
    return_tensors=None
    )

tokenizer.decode(inputs)

# 增强解码，一对句子
inputs = tokenizer.encode_plus(
    text=sents[0],
    text_pair=sents[1],  # None
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

# 增强解码，一对句子
inputs = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],  # [(sents[0], sents[1]), (sents[2], sents[3])]
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

class Dataset(th.utils.data.Dataset):
    def __init__(self):
        self.dataset = load_dataset(
            path="csv",
            data_files=os.path.join(path_data, "data.csv"),
            split="all"
            )
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        tokens = self.dataset[i]["tokens"]
        labels = self.dataset[i]["ner_tags"]
        return tokens, labels 

# ----------------------------------------------------------------------------------------------------------------
# 数据加载器
max_length = 50

def collate_fn(dataset):
    sents = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                         add_special_tokens=True,
                                         truncation=True,
                                         padding="max_length",
                                         max_length=max_length,
                                         return_tensors="pt",
                                         return_token_type_ids=True,
                                         return_attention_mask=True,
                                         return_special_tokens_mask=True,
                                         return_length=True,
                                         is_split_into_words=True)
    
    # 补齐label序列，BER要求
    for i in range(len(labels)):
        labels[i] = [7] + labels[i] + [7]*max_length
        labels[i] = labels[i][0:max_length]
    
    labels = th.LongTensor(labels)
    return inputs, labels

dataset = Dataset()
loader = th.utils.data.DataLoader(dataset=dataset,
                                  batch_size=16,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)

# ----------------------------------------------------------------------------------------------------------------
# 定义下游模型 Ner
class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.pretrained = config.get("pretrained")
        self.tagset_size = config.get("tagset_size")
        self.embedding_dim = config.get("embedding_dim")
        self.hidden_dim = config.get("hidden_dim")
        self.dropout = config.get("dropout")
        self.device = config.get("device")
        
        self.lstm_layer = nn.LSTM(input_size=self.embedding_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=1,
                                  dropout=self.dropout,
                                  batch_first=True,
                                  bidirectional=True)
        
        self.mlp_layer = nn.Linear(in_features=self.hidden_dim*2,
                                   out_features=self.tagset_size)
        
        self.crf = CRF(num_tags=self.tagset_size,
                       batch_first=True)
        
        def forward(self, inputs):
            # embedding_layer
            tokens = inputs["input_ids"]
            segments = inputs["token_type_ids"]
            valid_lens = inputs["attention_mask"]
            
            output_bert = self.pretrained(input_ids=tokens,
                                          token_type_ids=segments,
                                          attention_mask=valid_lens
                                          ).last_hidden_state
            
            # lstm_layer [num_layers, batch_size, hidden_dim]
            h0 = (th.randn([2, output_bert.shape[0], self.hidden_dim]) * 0.01).to(self.device)
            c0 = (th.randn([2, output_bert.shape[0], self.hidden_dim]) * 0.01).to(self.device)
            output_lstm, [ht, ct] = self.lstm_layer(output_bert, [h0, c0])
            
            # mlp_layer
            output_mlp = self.mlp_layer(output_lstm)
            return output_mlp
        
        def objt(self, output_mlp, labels):
            likelihood = self.crf(output_mlp, labels)
            loss = -1.0 * likelihood
            return loss
        
        def decode(self, inputs):
            output_mlp = self.forward(inputs)
            y_hat = self.crf.decode(output_mlp)
            return y_hat

# ----------------------------------------------------------------------------------------------------------------
# 训练
model = BERT_BiLSTM_CRF(config).to(device)
opti = optim.Adam(params=model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
# opti = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
epochs = 200

for epoch in range(epochs):
    # train
    loss_tmp = 0
    model.train()
    for (i, (inputs, labels)) in enumerate(loader):
        output_mlp = model(inputs)
        loss = model.objt(output_mlp, labels)
        loss_tmp += loss.item()
        
        opti.zero_grad()
        loss.backward()
        opti.step()
    
    loss_train = loss_tmp / (i+1)
    print(f"epoch {epoch}  loss_train {loss_train:.4f}")
    
# ----------------------------------------------------------------------------------------------------------------
# 测试

# ----------------------------------------------------------------------------------------------------------------
# 推理
query = ""

inputs_inference = tokenizer.batch_encode_plus(batch_text_or_text_pairs=[query],
                                               add_special_tokens=True,
                                               truncation=True,
                                               padding="max_length",
                                               max_length=max_length,
                                               return_tensors="pt",
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_special_tokens_mask=True,
                                               return_length=True,
                                               is_split_into_words=True)

tokenizer.decode(inputs_inference["input_ids"][0])

idx_to_tag = {}

model.eval()
with th.no_grad():
    y_hat = model.decode(inputs_inference)
    y_pred = np.array(y_hat).reshape(1, -1)[0][1:]
    y_pred = [idx_to_tag.get(y) for y in y_pred]















