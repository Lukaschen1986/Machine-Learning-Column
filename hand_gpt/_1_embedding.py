# -*- coding: utf-8 -*-
"""
https://www.bilibili.com/video/BV1Gf421Z75D/?spm_id_from=333.880.my_history.page.click&vd_source=fac9279bd4e33309b405d472b24286a8
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
from _1_1_token_embedding import TokenEmbedding
from _1_2_positional_embedding import PositionalEmbedding


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
class Embedding(nn.Module):
    def __init__(self, vocab_size, n_embd, valid_lens, dropout):
        super(Embedding, self).__init__()
        self.tok_embd = TokenEmbedding(vocab_size, n_embd)
        self.pos_embd = PositionalEmbedding(n_embd, valid_lens)
        self.drop_out = nn.Dropout(dropout)
    
    def forward(self, x):
        tok_embd = self.tok_embd(x)
        pos_embd = self.pos_embd(x)
        return self.drop_out(tok_embd + pos_embd)

