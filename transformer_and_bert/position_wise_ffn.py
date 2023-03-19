# -*- coding: utf-8 -*-
import os
import torch as th
from torch import nn
# from d2l import torch as d2l

th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    """
    ffn_num_input = 4; ffn_num_hiddens = 4; ffn_num_outputs = 8
    self = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_outputs)
    """
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense_1 = nn.Linear(ffn_num_input, ffn_num_hiddens, bias=True)
        self.relu = nn.ReLU()
        self.dense_2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs, bias=True)
    
    
    def forward(self, x):
        x = self.dense_1(x)  # torch.Size([2, 3, 4])
        x = self.relu(x)
        output_ffn = self.dense_2(x)  # torch.Size([2, 3, 8])
        return output_ffn
    


if __name__ == "__main__":
    ffn_num_input = 4
    ffn_num_hiddens = 4
    ffn_num_outputs = 8
    x = th.ones([2, 3, 4])
    
    ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_outputs)
    ffn.eval()
    
    output_ffn = ffn(x)
    
    
