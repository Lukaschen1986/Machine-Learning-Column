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
class FFN(nn.Module):
    def __init__(self, n_embd, n_hddn, dropout=0.1):
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_hddn, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hddn, n_embd, bias=True)
            )
    
    def forward(self, x):
        return self.mlp(x)

