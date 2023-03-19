# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from causal_self_attention import CausalSelfAttention
from new_gelu import NewGELU


class Block(nn.Module):
    """ 
    an unassuming Transformer block 
    self = Block(config)
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config["n_embd"], 4*config["n_embd"]),
            c_proj  = nn.Linear(4*config["n_embd"], config["n_embd"]),
            act     = NewGELU(),
            dropout = nn.Dropout(config["resid_pdrop"]),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        '''
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        '''
        x = self.ln_1(x)  # torch.Size([2, 4, 8])
        x = self.attn(x)  # torch.Size([2, 4, 8])
        x = self.ln_2(x)  # torch.Size([2, 4, 8])
        x = self.mlpf(x)  # torch.Size([2, 4, 8])
        return x
    

if __name__ == "__main__":
    config = {
        "model_type": "gpt2",
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 8,
        "vocab_size": 50257,
        "block_size": 1024,
        "embd_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "attn_pdrop": 0.1
        }
    
    x = torch.randn([2, 4, 8])
    block = Block(config)
    y = block(x)
    print(y.shape)  # torch.Size([2, 4, 8])
    