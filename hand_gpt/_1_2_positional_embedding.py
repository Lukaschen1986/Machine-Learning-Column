# -*- coding: utf-8 -*-
"""
https://www.bilibili.com/video/BV1Gf421Z75D/?spm_id_from=333.880.my_history.page.click&vd_source=fac9279bd4e33309b405d472b24286a8
"""
import os
import warnings; warnings.filterwarnings("ignore")
import torch as th
import torch.nn as nn


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
class PositionalEmbedding(nn.Module):
    def __init__(self, n_embd, valid_lens):
        super(PositionalEmbedding, self).__init__()
        self.encoding = th.zeros([valid_lens, n_embd])
        self.encoding.requires_grad_(False)
        
        pos = th.arange(0, valid_lens)
        pos = pos.float().unsqueeze(dim=1)
        _2i = th.arange(0, n_embd, 2)
        
        self.encoding[:, 0::2] = th.sin(pos / (10000**(_2i/n_embd)))
        self.encoding[:, 1::2] = th.cos(pos / (10000**(_2i/n_embd)))
    
    def forward(self, x):
        seq_lens = x.shape[1]
        return self.encoding[:seq_lens, :]
        
    

