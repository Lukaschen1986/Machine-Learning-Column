# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    

if __name__ == "__main__":
    x = torch.randn([2, 4, 8])
    act = NewGELU()
    y = act(x)
    print(y.shape)  # torch.Size([2, 4, 8])
