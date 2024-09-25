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
class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=10**-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(th.ones(n_embd))
        self.beta = nn.Parameter(th.zeros(n_embd))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)  # unbiased=False 总体方差
        out = (x - mean) / th.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
        
    
