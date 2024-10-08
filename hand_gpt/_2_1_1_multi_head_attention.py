# -*- coding: utf-8 -*-
"""
https://www.bilibili.com/video/BV1o2421A7Dr/?spm_id_from=333.788&vd_source=fac9279bd4e33309b405d472b24286a8
https://dwexzknzsh8.feishu.cn/docx/VkYud3H0zoDTrrxNX5lce0S4nDh
"""
import os
import sys
import warnings; warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------
device = th.device("cuda" if th.cuda.is_available() else "cpu")
devive_cnt = th.cuda.device_count()
print(f"device = {device}; devive_cnt = {devive_cnt}")
print(th.__version__)
print(th.version.cuda)

# ----------------------------------------------------------------------------------------------------------------
path_project = os.getcwd()
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")
path_output = os.path.join(os.path.dirname(path_project), "output")

# ----------------------------------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_embd_split = self.n_embd // self.n_head
        self.w_q = nn.Linear(n_embd, n_embd, bias=True)
        self.w_k = nn.Linear(n_embd, n_embd, bias=True)
        self.w_v = nn.Linear(n_embd, n_embd, bias=True)
        self.w_o = nn.Linear(n_embd, n_embd, bias=True)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=False):
        batch_size, valid_lens, q_embd = q.shape
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        q = q.reshape(batch_size, valid_lens, self.n_head, self.n_embd_split).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, valid_lens, self.n_head, self.n_embd_split).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, valid_lens, self.n_head, self.n_embd_split).permute(0, 2, 1, 3)
        
        score = q @ k.transpose(2, 3) / math.sqrt(self.n_embd_split)
        if mask:
            # mask = th.tril(th.ones([valid_lens, valid_lens]))
            score = score.masked_fill(mask == 0, float("-inf"))
        score = self.softmax(score) @ v
        
        score = score.permute(0, 2, 1, 3).reshape(batch_size, valid_lens, q_embd)  # .reshape == .contiguous().view()
        output = self.w_o(score)
        return output
        

class GroupQueryAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_group):
        super(GroupQueryAttention, self).__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_group = n_group
        
        assert n_embd % n_head == 0
        self.n_embd_split = self.n_embd // self.n_head  # 每个头分多少维度
        self.n_head_group = self.n_head // self.n_group  # 每个组分多少头
        
        self.w_q = nn.Linear(n_embd, n_embd, bias=True)
        self.w_k = nn.Linear(n_embd, self.n_group * self.n_embd_split, bias=True)
        self.w_v = nn.Linear(n_embd, self.n_group * self.n_embd_split, bias=True)
        self.w_o = nn.Linear(n_embd, n_embd, bias=True)
        self.softmax = nn.Softmax(dim=-1)
    
    def expand(self, x):
        batch_size, valid_lens, _ = x.shape
        x = x[:, :, None, :, :].expand(batch_size, self.n_group, self.n_head_group, valid_lens, self.n_embd)
        return x.view(batch_size, self.n_group * self.n_head_group, valid_lens, self.n_embd)
        
    def forward(self, q, k, v, mask=False):
        batch_size, valid_lens, q_embd = q.shape
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        q = q.reshape(batch_size, valid_lens, self.n_head, self.n_embd_split).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, valid_lens, self.n_group, self.n_embd_split).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, valid_lens, self.n_group, self.n_embd_split).permute(0, 2, 1, 3)
        
        k, v = self.expand(k), self.expand(v)
        score = q @ k.transpose(2, 3) / math.sqrt(self.n_embd_split)
        if mask:
            score = score.masked_fill(mask == 0, float("-inf"))
        score = self.softmax(score) @ v
        
        score = score.permute(0, 2, 1, 3).contiguous().reshape(batch_size, valid_lens, q_embd)
        output = self.w_o(score)
        return output


if __name__ == "__main__":
    n_embd = 768
    n_head = 8
    attention = MultHeadAttention(n_embd, n_head)
    x = th.randn(2, 16, n_embd)  # batch_size, valid_lens, n_embd
    output = attention(x, x, x)
    output.shape
    
    
    
