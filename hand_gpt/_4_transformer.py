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
    def __init__(self, source_pad_idx, target_pad_idx, enc_vocab_size, dec_vocab_size, 
                 valid_lens, n_embd, n_head, n_hddn, n_layer, dropout):
        