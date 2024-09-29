import os
import sys
import warnings; warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from _1_embedding import Embedding
from _2_1_encoder_layer import EncoderLayer


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
class Encoder(nn.Module):
    def __init__(self, vocab_size, valid_lens, n_embd, n_head, n_hddn, n_layer, dropout):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, n_embd, valid_lens, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(n_embd, n_head, n_hddn, dropout) for _ in range(n_layer)]
            )
        
    def forward(self, x_enc, padd_mask):
        x_enc = self.embedding(x_enc)
        for layer in self.layers:
            x_enc = layer(x_enc, padd_mask)
        return x_enc