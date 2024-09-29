# -*- coding: utf-8 -*-
"""
https://www.bilibili.com/video/BV1oK421Y7Vh/?spm_id_from=333.788&vd_source=fac9279bd4e33309b405d472b24286a8
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
from _2_encoder import Encoder
from _3_decoder import Decoder


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
class Transformer(nn.Module):
    def __init__(self, enc_pad_idx, dec_pad_idx, enc_vocab_size, dec_vocab_size, 
                 valid_lens, n_embd, n_head, n_hddn, n_layer, dropout):
        self.encoder = Encoder(enc_vocab_size, valid_lens, n_embd, n_head, n_hddn, n_layer, dropout)
        self.decoder = Decoder(dec_vocab_size, valid_lens, n_embd, n_head, n_hddn, n_layer, dropout)
        self.enc_pad_idx = enc_pad_idx
        self.dec_pad_idx = dec_pad_idx
    
    def get_time_mask(self, q, k):
        q_lens, k_lens = q.size(1), k.size(1)
        time_mask = th.tril(th.ones([q_lens, k_lens]))  #.type(th.BoolTensor)
        return time_mask
    
    def get_padd_mask(self, q, k, q_pad_idx, k_pad_idx):
        q_lens, k_lens = q.size(1), k.size(1)
        
        # (batch_size, valid_lens, q_lens, k_lens)
        q_mask = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        q_mask = q_mask.repeat(1, 1, 1, k_lens)
        
        k_mask = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        k_mask = k_mask.repeat(1, 1, q_lens, 1)
        
        padd_mask = q_mask & k_mask
        return padd_mask
    
    def forward(self, x_enc, x_dec):
        padd_mask = self.get_padd_mask(x_enc, x_enc, self.enc_pad_idx, self.enc_pad_idx)
        time_mask = self.get_padd_mask(x_dec, x_dec, self.dec_pad_idx, self.dec_pad_idx) * \
            self.get_time_mask(x_dec, x_dec)
        padd_time_mask = self.get_padd_mask(x_dec, x_enc, self.dec_pad_idx, self.enc_pad_idx)
        
        x_enc = self.encoder(x_enc, padd_mask)
        x_dec = self.decoder(x_dec, x_enc, time_mask, padd_time_mask)
        return x_dec

