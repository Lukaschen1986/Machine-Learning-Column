# -*- coding: utf-8 -*-
'''
https://zhuanlan.zhihu.com/p/498677758

GPT-2能够处理多种自然语言处理任务，包括但不限于：

文本生成：根据给定的上下文生成连续、有意义的文本。
文本分类：根据给定的文本对其进行分类。
问答系统：回答用户提出的问题。
词汇补全：自动补全词语或句子。
情感分析：分析文本中的情感（如积极或消极）。
机器翻译：将一种语言的文本翻译成另一种语言的文本。
文本摘要：从长篇文本中提取出关键信息，生成较短的摘要。

总的来说，GPT-2是目前自然语言处理领域最强大和最先进的模型之一，可用于许多自然语言处理任务。
'''
import warnings; warnings.filterwarnings("ignore")
import os
# from itertools import zip_longest
import numpy as np
import torch as th
from torch import nn
from transformers import (BertTokenizer, AutoTokenizer, BertModel, AdamW,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          BertTokenizer, GPT2Model)
from datasets import (load_dataset, load_from_disk)
import torch.optim as optim
# from rouge import Rouge


device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/hugging_face"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 加载分词器
# checkpoint = "bert-base-chinese"
# checkpoint = "hfl/rbt6"
# checkpoint = "hfl/chinese-bert-wwm-ext"
# checkpoint = "nghuyong/ernie-3.0-base-zh"
# checkpoint = "nghuyong/ernie-1.0-base-zh"
checkpoint = "uer/gpt2-chinese-cluecorpussmall"
'''
由于GPT2-Chinese项目使用的tokenizer是BertTokenizer，所以这里载入也适用BertTokenizer，保证对齐。
载入模型之后就可以进行使用和预测了
'''
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
)

# ----------------------------------------------------------------------------------------------------------------
# 加载预训练模型
pretrained = GPT2LMHeadModel.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=path_model,
    force_download=False,
    local_files_only=False
    )

# for param in pretrained.parameters():
#     param.requires_grad_(False)

# ----------------------------------------------------------------------------------------------------------------
inputs_text = "你说"
input_ids = tokenizer.encode(inputs_text)
tokenizer.decode(input_ids)  # '[CLS] 你 说 [SEP]'

input_ids = input_ids[:-1]
inputs = {"input_ids": th.tensor([input_ids])}
outputs = pretrained(**inputs, labels=th.tensor([input_ids]))

outputs.loss  # tensor(4.9737, grad_fn=<NllLossBackward0>)
outputs.logits.shape  # torch.Size([1, 3, 21128])
'''
encode方法会把input_text变成id，但这里会自动把text首尾加上"[CLS]"和"[SEP]"，
也就是input_text的"你说"会变成"[CLS]你说[SEP]"，
在GPT2中推理的时候，首是要加上"[CLS]"的，这个是必须的。
但尾的那个 "[SEP]"要去掉，因为当前预训练模型数据格式"[SEP]"是终止符，
input_ids[:-1]就是把"[SEP]"去掉。
'''

next_token_id = int(np.argmax(outputs.logits[0][-1].detach().numpy()))
next_token = tokenizer.convert_ids_to_tokens(next_token_id)
print(next_token)  # 过
print(inputs_text + next_token)  # 你说过

# ----------------------------------------------------------------------------------------------------------------
# 定义数据集
class Dataset(th.utils.data.Dataset):
    def __init__(self, split):
        if split == "train":
            self.dataset = load_dataset(
                path="csv",
                data_files=os.path.join(path_data, "gpt/gpt_train.csv"),
                split="all"
                )
        elif split == "test":
            self.dataset = load_dataset(
                path="csv",
                data_files=os.path.join(path_data, "gpt/gpt_test.csv"),
                split="all"
                )
        else:
            raise ValueError
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        question = self.dataset[i]["question"]
        answer = self.dataset[i]["answer"]
        return question, answer


dataset = Dataset(split="train")

for (q, a) in dataset:
    break
max(len(q) + len(a) for (q, a) in dataset)

# ----------------------------------------------------------------------------------------------------------------
# 整理函数
def collate_fn(dataset):
    questions = [x[0] for x in dataset]
    answers = [x[1] for x in dataset]
    max_length = max(len(q) + len(a) for (q, a) in dataset)
    
    #编码
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=zip(questions, answers),
                                         max_length=max_length,
                                         padding="max_length",
                                         add_special_tokens=True,
                                         truncation=True,
                                         return_tensors="pt",
                                         return_token_type_ids=True,
                                         return_attention_mask=True,
                                         return_special_tokens_mask=True,
                                         return_length=True,
                                         is_split_into_words=False)
    
    '''
    tokenizer.decode(inputs["input_ids"][0])
    '[CLS] 新 款 植 绒 波 点 飘 带 灯 笼 袖 衬 衫 [SEP] 灯 笼 袖 衬 衫 ， 
    浪 漫 斐 然 显 优 雅 [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
    [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'
    '''
    return inputs

# def collate_fn(batch):
#     input_ids = []
#     input_lens_list = [len(w) for w in batch]
#     max_input_len = max(input_lens_list)
    
#     for i in range(len(batch)):
#         input_len = len(batch[i])
#         input_ids.append(batch[i])
#         input_ids[i].extend([tokenizer.pad_token_id] * (max_input_len - input_len))
#     return th.tensor(input_ids, dtype=torch.long)

# 数据迭代器
batch_size = 16
loader_train = th.utils.data.DataLoader(dataset=Dataset(split="train"),
                                        batch_size=batch_size,
                                        collate_fn=collate_fn,
                                        shuffle=True,
                                        drop_last=False)

# ----------------------------------------------------------------------------------------------------------------
# 定义下游模型
config = {
    "pretrained": pretrained,
    "device": device
    }

class Model(th.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pretrained = config.get("pretrained")
        self.device = config.get("device")

    def forward(self, inputs):
        tokens = inputs["input_ids"].to(self.device)
        segments = inputs["token_type_ids"].to(self.device)
        valid_lens = inputs["attention_mask"].to(self.device)
        
        output_gpt = self.pretrained(
            input_ids=tokens,
            token_type_ids=segments,
            attention_mask=valid_lens
            ).logits
        
        return output_gpt


# ----------------------------------------------------------------------------------------------------------------
# 训练
model = Model(config).to(device)
opti = optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
# opti = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
objt = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="mean")

epochs = 100
loss_res = []

for epoch in range(epochs):
    # train
    loss_tmp = 0
    model.train()
    for (i, inputs) in enumerate(loader_train):
        # break
        # tokenizer.decode(inputs["input_ids"][0])
        
        tokens = inputs["input_ids"].to(device)
        output_gpt = model(inputs)
        
        shift_logits = output_gpt[:, 0:-1, :].flatten(start_dim=0, end_dim=1)
        shift_labels = tokens[:, 1:].flatten(start_dim=0, end_dim=1)
        
        # loss
        loss = objt(shift_logits, shift_labels)
        loss_tmp += loss.item()
        
        # accuracy
        # y_hat, y_pred = shift_logits.max(dim=-1)
        # not_ignore = th.ne(shift_labels, tokenizer.pad_token_id)
        # num_targets = not_ignore.long().sum().item()
        
        # correct = (shift_labels == y_pred) & not_ignore
        # correct = correct.float().sum()
        # accu = correct / num_targets
        
        opti.zero_grad()
        loss.backward()
        opti.step()
    
    loss_train = loss_tmp / (i+1)
    print(f"epoch {epoch}  loss_train {loss_train:.4f}")
    loss_res.append(loss_train)
    
# ----------------------------------------------------------------------------------------------------------------
# 测试
loader_test = th.utils.data.DataLoader(dataset=Dataset(split="test"),
                                       batch_size=50,
                                       collate_fn=collate_fn,
                                       shuffle=False,
                                       drop_last=False)

loss_tmp = 0
model.eval()
with th.no_grad():
    for (i, inputs) in enumerate(loader_test):
        tokens = inputs["input_ids"].to(device)
        output_gpt = model(inputs)
        
        shift_logits = output_gpt[:, 0:-1, :].flatten(start_dim=0, end_dim=1)
        shift_labels = tokens[:, 1:].flatten(start_dim=0, end_dim=1)
        
        # loss
        loss = objt(shift_logits, shift_labels)
        loss_tmp += loss.item()
    
    loss_test = loss_tmp / (i+1)
        
# ----------------------------------------------------------------------------------------------------------------
# 推理
'''
对于文案生成任务，输入要处理成"[CLS]商品标题[SEP]商品文案[SEP]"的格式，
这样finetune好gpt2模型之后，预测时输入"[CLS]商品标题[SEP]"即可进行文案生成，
生成到"[SEP]"字符或者最大长度时，停止解码。

在GPT2中推理的时候，首是要加上"[CLS]"的，这个是必须的。
第一个 "[SEP]"保留，意味着从这个[SEP]开始推理。
'''
question = "亲肤人棉花卉印花睡袍"
max_length = 100
answer = ""

inputs_init = tokenizer.batch_encode_plus(batch_text_or_text_pairs=[question],
                                          max_length=max_length,
                                          add_special_tokens=True,
                                          truncation=True,
                                          padding="do_not_pad",
                                          return_tensors=None,
                                          return_token_type_ids=True,
                                          return_attention_mask=True,
                                          return_special_tokens_mask=True,
                                          return_length=False,
                                          is_split_into_words=False
                                          )
inputs_init
'''
{
 'input_ids': [[101, 3173, 3621, 3490, 5309, 3797, 4157, 7603, 2372, 4128, 5021, 6153, 6137, 6136, 102]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'special_tokens_mask': [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
'''
tokenizer.decode(inputs_init["input_ids"][0])

tokens = inputs_init["input_ids"][0]
segments = inputs_init["token_type_ids"][0]
valid_lens = inputs_init["attention_mask"][0]

model.eval()
with th.no_grad():
    for i in range(max_length):
        inputs_update = {
            "input_ids": th.tensor([tokens]),
            "token_type_ids": th.tensor([segments]),
            "attention_mask": th.tensor([valid_lens])
            }
        output_gpt = model(inputs_update)
        
        last_token_id = int(output_gpt[0][-1].cpu().detach().numpy().argmax())
        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        answer += last_token
        
        if last_token_id == tokenizer.sep_token_id:
            break
        else:
            tokens.append(last_token_id)
            segments.append(1)
            valid_lens.append(1)
        
print(answer)
