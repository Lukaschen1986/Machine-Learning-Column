# -*- coding: utf-8 -*-
import os
import torch as th
from torch import nn
# from d2l import torch as d2l

th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# 位置编码
class PositionEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        fenzi = th.arange(max_len, dtype=th.float64).reshape(-1, 1)  # torch.Size([1000, 1])
        fenmu = 10000**(th.arange(0, num_hiddens, 2, dtype=th.float64) / num_hiddens)  # torch.Size([16])
        data = fenzi / fenmu  # torch.Size([1000, 16])
        
        self.P = th.zeros([1, max_len, num_hiddens])  # torch.Size([1, 1000, 32])
        self.P[:, :, 0::2] = th.sin(data)  # 偶数列
        self.P[:, :, 1::2] = th.cos(data)  # 奇数列
    
    
    def forward(self, x):
        output_pos = x + self.P[:, 0:x.shape[1], :].to(x.device)
        output_pos = self.dropout(output_pos)
        return output_pos



if __name__ == "__main__":
    num_hiddens = 4  # 隐藏层维度
    num_steps = 60  # 序列长度
    dropout = 0.5
    
    x = th.zeros([1, num_steps, num_hiddens])  # torch.Size([1, 60, 32])
    
    pos_encoding = PositionEncoding(num_hiddens, dropout)
    pos_encoding.eval()
    
    output_pos = pos_encoding(x)
    output_pos.shape  # torch.Size([1, 60, 32])
    
    # 可视化
    # X = pos_encoding(th.zeros((1, num_steps, encoding_dim)))
    # P = pos_encoding.P[:, :X.shape[1], :]
    # d2l.plot(th.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
    #          figsize=(6, 2.5), legend=["Col %d" % d for d in th.arange(6, 10)])