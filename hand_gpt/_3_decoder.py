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
from _3_1_decoder_layer import DecoderLayer


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
class Decoder(nn.Module):
    def __init__(self, vocab_size, valid_lens, n_embd, n_head, n_hddn, n_layer, dropout):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, n_embd, valid_lens, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(n_embd, n_head, n_hddn, dropout) for _ in range(n_layer)]
            )
        self.fc = nn.Linear(n_embd, vocab_size, bias=True)
        
    def forward(self, x_dec, x_enc, time_mask, padd_mask):
        x_dec = self.embedding(x_dec)
        for layer in self.layers:
            x_dec = layer(x_dec, x_enc, time_mask, padd_mask)
        x_dec = self.fc(x_dec)
        return x_dec
    
    