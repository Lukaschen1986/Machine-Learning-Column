# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
# from mingpt.utils import CfgNode as CN


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    
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
    self = CausalSelfAttention(config)
    """
    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config["n_embd"], 3*config["n_embd"])
        # output projection
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"])
        # regularization
        self.attn_dropout = nn.Dropout(config["attn_pdrop"])
        self.resid_dropout = nn.Dropout(config["resid_pdrop"])
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config["block_size"], config["block_size"]))
                                     .view(1, 1, config["block_size"], config["block_size"]))
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        '''
        self.c_attn(x).shape  # torch.Size([2, 4, 24])
        
        q.shape  # torch.Size([2, 4, 8])
        k.shape  # torch.Size([2, 4, 8])
        v.shape  # torch.Size([2, 4, 8])
        '''
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        '''
        q.shape  # torch.Size([2, 4, 4, 4])
        k.shape  # torch.Size([2, 4, 4, 4])
        v.shape  # torch.Size([2, 4, 4, 4])
        '''
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        '''
        att.shape  # torch.Size([2, 2, 4, 4])
        '''
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        '''
        y.shape  # torch.Size([2, 4, 8])
        '''
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        '''
        y.shape  # torch.Size([2, 4, 8])
        '''
        return y


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
    attn = CausalSelfAttention(config)
    y = attn(x)
    print(y.shape)  # torch.Size([2, 4, 8])
