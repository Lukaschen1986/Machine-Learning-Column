# -*- coding: utf-8 -*-
import os
import torch as th
from torch import nn
from transformers import (BertTokenizer, AutoTokenizer, BertModel, AdamW)
from datasets import (load_dataset, load_from_disk)


# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 加载分词器
pretrained_model_name = "bert-base-chinese"
pretrained_model_name = "hfl/rbt6"
pretrained_model_name = "hfl/chinese-bert-wwm-ext"
pretrained_model_name = "nghuyong/ernie-3.0-base-zh"
pretrained_model_name = "nghuyong/ernie-1.0-base-zh"

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name,
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
)

# tokens = tokenizer.encode(
#     text=sents[0],
#     text_pair=sents[1],
#     truncation=True,
#     padding="max_length",
#     add_special_tokens=True,
#     max_length=30,
#     return_tensors=None
#     )

# tokens = tokenizer.encode_plus(
#     text=sents[0],
#     text_pair=sents[1],
#     truncation=True,
#     padding="max_length",
#     max_length=30,
#     add_special_tokens=True,
#     return_tensors=None,
#     return_token_type_ids=True,
#     return_attention_mask=True,
#     return_special_tokens_mask=True,
#     return_length=True
#     )

# tokens = tokenizer.batch_encode_plus(
#     batch_text_or_text_pairs=[sents[0], sents[1]],  # [(sents[0], sents[1]), (sents[2], sents[3])]
#     add_special_tokens=True,
#     truncation=True,
#     padding="max_length",
#     max_length=15,
#     return_tensors=None,
#     return_token_type_ids=True,
#     return_attention_mask=True,
#     return_special_tokens_mask=True,
#     return_length=True,
#     is_split_into_words=True
#     )

# zidian = tokenizer.get_vocab()
# tokenizer.add_tokens(new_tokens=['月光', '希望'])
# tokenizer.add_special_tokens({'eos_token': '[EOS]'})

# ----------------------------------------------------------------------------------------------------------------
# 定义数据集
# names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

# dataset = load_dataset(
#     path="seamew/ChnSentiCorp",
#     cache_dir=path_data
#     )

class Dataset(th.utils.data.Dataset):
    def __init__(self):
        dataset = load_from_disk(dataset_path=os.path.join(path_data, "peoples_daily_ner"))
        # dataset = load_dataset(
        #     path="csv",
        #     data_files=os.path.join(path_data, "peoples_daily_ner"),
        #     cache_dir=path_data
        #     )
        # dataset_split = dataset.train_test_split(test_size=0.1)
        
        def func(data):
            return len(data["tokens"]) <= 512 - 2
        self.dataset = dataset.filter(func)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        tokens = self.dataset[i]["tokens"]
        labels = self.dataset[i]["ner_tags"]
        return tokens, labels 

print(dataset["train"][0])
'''
{'id': '0', 'tokens': ['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', 
                       '间', '的', '海', '域', '。'], 
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]}
'''

# ----------------------------------------------------------------------------------------------------------------
# 数据加载器
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                         add_special_tokens=True,
                                         truncation=True,
                                         padding="max_length",
                                         max_length=,
                                         return_tensors="pt",
                                         return_token_type_ids=True,
                                         return_attention_mask=True,
                                         return_special_tokens_mask=True,
                                         return_length=True,
                                         is_split_into_words=True)
    
    # max_lens = inputs['input_ids'].shape[1]
    
    # for i in range(len(labels)):
    #     labels[i] = [7] + labels[i]
    #     labels[i] += [7] * max_lens
    #     labels[i] = labels[i][:max_lens]
    
    # tokens = data['input_ids']
    # segmens = data['token_type_ids']
    # valid_lens = data['attention_mask']
    # labels = torch.LongTensor(labels)
    return inputs, labels

loader = th.utils.data.DataLoader(dataset=dataset,
                                  batch_size=16,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)


# ----------------------------------------------------------------------------------------------------------------
# 加载预训练模型
pretrained = BertModel.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name,
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
    )

# ----------------------------------------------------------------------------------------------------------------
# 定义下游模型
# 考虑用 Bert-BiLSTM-CRF 改写
class NextModel(nn.Module):
    def __init__(self):
        super(NextModel).__init__()
        self.tuneing = False
        self.pretrained = None
        self.rnn = torch.nn.GRU(768, 768, batch_first=True)
        self.fc = torch.nn.Linear(768, 8)
    
    def forward(self, inputs):
        if self.tuneing:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state

        out, _ = self.rnn(out)
        out = self.fc(out).softmax(dim=2)
        return out

    def fine_tuneing(self, tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in pretrained.parameters():
                i.requires_grad = True

            pretrained.train()
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)

            pretrained.eval()
            self.pretrained = None

# ----------------------------------------------------------------------------------------------------------------
# 对计算结果和label变形,并且移除pad
def reshape_and_remove_pad(outs, labels, attention_mask):
    #变形,便于计算loss
    #[b, lens, 8] -> [b*lens, 8]
    outs = outs.reshape(-1, 8)
    #[b, lens] -> [b*lens]
    labels = labels.reshape(-1)

    #忽略对pad的计算结果
    #[b, lens] -> [b*lens - pad]
    select = attention_mask.reshape(-1) == 1
    outs = outs[select]
    labels = labels[select]
    return outs, labels


# 获取正确数量和总数
def get_correct_and_total_count(labels, outs):
    #[b*lens, 8] -> [b*lens]
    outs = outs.argmax(dim=1)
    correct = (outs == labels).sum().item()
    total = len(labels)

    #计算除了0以外元素的正确率,因为0太多了,包括的话,正确率很容易虚高
    select = labels != 0
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs == labels).sum().item()
    total_content = len(labels)
    return correct, total, correct_content, total_content

# ----------------------------------------------------------------------------------------------------------------
# 训练
lr = 2e-5 if model.tuneing else 5e-4
optimizer = AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
epochs = 10

model.train()
for epoch in range(epochs):
    for step, (inputs, labels) in enumerate(loader):
        #模型计算
        #[b, lens] -> [b, lens, 8]
        outs = model(inputs)

        #对outs和label变形,并且移除pad
        #outs -> [b, lens, 8] -> [c, 8]
        #labels -> [b, lens] -> [c]
        outs, labels = reshape_and_remove_pad(outs, labels,
                                              inputs['attention_mask'])

        #梯度下降
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            counts = get_correct_and_total_count(labels, outs)

            accuracy = counts[0] / counts[1]
            accuracy_content = counts[2] / counts[3]

            print(epoch, step, loss.item(), accuracy, accuracy_content)

    torch.save(model, 'model/命名实体识别_中文.model')


model.fine_tuneing(False)
print(sum(p.numel() for p in model.parameters()) / 10000)

# ----------------------------------------------------------------------------------------------------------------
# 测试
model_load = torch.load('model/命名实体识别_中文.model')
model_load.eval()

loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                          batch_size=128,
                                          collate_fn=collate_fn,
                                          shuffle=True,
                                          drop_last=True)

correct = 0
total = 0

correct_content = 0
total_content = 0

for step, (inputs, labels) in enumerate(loader_test):
    if step == 5:
        break
    print(step)

    with torch.no_grad():
        #[b, lens] -> [b, lens, 8] -> [b, lens]
        outs = model_load(inputs)

    #对outs和label变形,并且移除pad
    #outs -> [b, lens, 8] -> [c, 8]
    #labels -> [b, lens] -> [c]
    outs, labels = reshape_and_remove_pad(outs, labels,
                                          inputs['attention_mask'])

    counts = get_correct_and_total_count(labels, outs)
    correct += counts[0]
    total += counts[1]
    correct_content += counts[2]
    total_content += counts[3]

print(correct / total, correct_content / total_content)

# ----------------------------------------------------------------------------------------------------------------
# 预测
model_load = torch.load('model/命名实体识别_中文.model')
model_load.eval()

loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                          batch_size=32,
                                          collate_fn=collate_fn,
                                          shuffle=True,
                                          drop_last=True)

for i, (inputs, labels) in enumerate(loader_test):
    break

with torch.no_grad():
    #[b, lens] -> [b, lens, 8] -> [b, lens]
    outs = model_load(inputs).argmax(dim=2)

for i in range(32):
    #移除pad
    select = inputs['attention_mask'][i] == 1
    input_id = inputs['input_ids'][i, select]
    out = outs[i, select]
    label = labels[i, select]
    
    #输出原句子
    print(tokenizer.decode(input_id).replace(' ', ''))

    #输出tag
    for tag in [label, out]:
        s = ''
        for j in range(len(tag)):
            if tag[j] == 0:
                s += '·'
                continue
            s += tokenizer.decode(input_id[j])
            s += str(tag[j].item())

        print(s)
    print('==========================')
