# -*- coding: utf-8 -*-
"""
https://www.bilibili.com/video/BV1dr421w7J5?p=1&vd_source=fac9279bd4e33309b405d472b24286a8
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
class LoraLayer1(nn.Module):
    """
    self = LoraLayer1(in_features=100, out_features=2, rank=8, lora_alpha=16, lora_dropout=0.1)
    """
    def __init__(self, in_features, out_features, rank=8, lora_alpha=16, lora_dropout=0.1):
        super(LoraLayer1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.linear = nn.Linear(self.in_features, self.out_features, bias=True)
        
        if self.rank <= 0:
            raise ValueError("rank must > 0")
            
        # init weight
        self.lora_a_weight = nn.init.xavier_normal_(
            nn.Parameter(th.zeros([self.rank, self.in_features]))
            )
        self.lora_b_weight = nn.Parameter(th.zeros([self.out_features, self.rank]))
        
        # use rslora
        self.scale = self.lora_alpha / math.sqrt(self.rank)
        
        # linear weight
        self.linear.weight.requires_grad_(False)
        
        # lora dropout
        if self.lora_dropout > 0:
            self.drop = nn.Dropout(self.lora_dropout)
        else:
            self.drop = nn.Identity()
        
    def forward(self, x):
        output = F.linear(input=x, 
                          weight=self.linear.weight + self.lora_b_weight @ self.lora_a_weight * self.scale, 
                          bias=self.linear.bias)  # h = (W + BA)x + b
        return self.drop(output)


class LoraLayer2(nn.Module):
    """
    self = LoraLayer2(in_features=100, out_features=2, rank=8, lora_alpha=16, lora_dropout=0.1)
    """
    def __init__(self, in_features, out_features, rank=8, lora_alpha=16, lora_dropout=0.1):
        super(LoraLayer2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.linear = nn.Linear(self.in_features, self.out_features, bias=True)
        
        if self.rank <= 0:
            raise ValueError("rank must > 0")
            
        # init layer
        self.lora_a = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, self.out_features, bias=False)
        
        nn.init.xavier_normal_(self.lora_a.weight)
        nn.init.zeros_(self.lora_b.weight)
        
        # use rslora
        self.scale = self.lora_alpha / math.sqrt(self.rank)
            
        # linear weight
        self.linear.weight.requires_grad_(False)
        
        # lora dropout
        if self.lora_dropout > 0:
            self.drop = nn.Dropout(self.lora_dropout)
        else:
            self.drop = nn.Identity()
        
    def forward(self, x):
        output_linr = self.linear(x)
        output_lora = self.lora_a(x)
        output_lora = self.lora_b(output_lora)
        output_lora *= self.scale
        output = output_linr + output_lora  # h = Wx+b + BAx
        return self.drop(output)
    


if __name__ == "__main__":
    x = th.randn(1000, 100)
    
    LoRA1 = LoraLayer1(in_features=100, out_features=2, rank=8, lora_alpha=16, lora_dropout=0.1)
    print(LoRA1)
    """
    LoraLayer1(
      (linear): Linear(in_features=100, out_features=2, bias=True)
      (drop): Dropout(p=0.1, inplace=False)
    )
    """
    
    LoRA2 = LoraLayer2(in_features=100, out_features=2, rank=8, lora_alpha=16, lora_dropout=0.1)
    print(LoRA2)
    """
    LoraLayer2(
      (linear): Linear(in_features=100, out_features=2, bias=True)
      (lora_a): Linear(in_features=100, out_features=8, bias=False)
      (lora_b): Linear(in_features=8, out_features=2, bias=False)
      (drop): Dropout(p=0.1, inplace=False)
    )
    """
    
    output1 = LoRA1(x)
    output2 = LoRA2(x)

