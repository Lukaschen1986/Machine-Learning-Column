# -*- coding: utf-8 -*-
"""
https://www.bilibili.com/video/BV1QA4m1P7w3/?spm_id_from=333.788&vd_source=fac9279bd4e33309b405d472b24286a8
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
from hand_gpt._2_1_1_multi_head_attention import MultiHeadAttention
from _2_1_2_layer_norm import LayerNorm
from _2_1_3_ffn import FFN


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
class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, n_hddn, dropout):
        super(DecoderLayer, self).__init__()
        self.attention = MultHeadAttention(n_embd, n_head)
        self.drop = nn.Dropout(dropout)
        self.norm = LayerNorm(n_embd)
        self.ffn = FFN(n_embd, n_hddn, dropout)
    
    def forward(self, x_dec, x_enc, time_mask, padd_mask):
        x_pre = x_dec.clone()
        x_dec = self.attention(x_dec, x_dec, x_dec, time_mask)  # 下三角掩码
        x_dec = self.drop(x_dec)
        x_dec = self.norm(x_dec + x_pre)
        
        if x_enc:
            x_pre = x_dec.clone()
            x_dec = self.attention(x_dec, x_enc, x_enc, padd_mask)  # padding 掩码
            x_dec = self.drop(x_dec)
            x_dec = self.norm(x_dec + x_pre)
        
        x_pre = x_dec.clone()
        x_dec = self.ffn(x_dec)
        x_dec = self.drop(x_dec)
        x_dec = self.norm(x_dec + x_pre)
        return x_dec




