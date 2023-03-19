# -*- coding: utf-8 -*-
import os
import torch as th
from torch import nn
from d2l import torch as d2l

th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# 多头注意力
class MultiHeadAttention(nn.Module):
    """
    self = MultiHeadAttention(query_size=num_hiddens, key_size=num_hiddens, value_size=num_hiddens, 
                              num_hiddens=num_hiddens, num_heads=num_heads, dropout=0.5, bias=False)
    queries=x; keys=y; values=y; valid_lens=valid_lens
    """
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, 
                 dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.attention = d2l.DotProductAttention(dropout)
    
    
    def transpose_qkv(self, x):
        """为了多注意力头的并行计算而变换形状"""
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        return x
    
    
    def transpose_output(self, x):
        """逆转transpose_qkv函数的操作"""
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x


    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))  # torch.Size([10, 4, 20])
        keys = self.transpose_qkv(self.W_k(keys))  # torch.Size([10, 6, 20])
        values = self.transpose_qkv(self.W_v(values))  # torch.Size([10, 6, 20])

        if valid_lens is not None:
            valid_lens = th.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            
        output = self.attention(queries, keys, values, valid_lens)  # torch.Size([10, 4, 20])
        output_concat = self.transpose_output(output)  # torch.Size([2, 4, 100])
        output_attn = self.W_o(output_concat)  # torch.Size([2, 4, 100])
        return output_attn



if __name__ == "__main__":
    num_hiddens = 100  # 隐藏层维度
    num_heads = 5
    batch_size = 2
    num_queries = 4  # 序列长度
    num_kvpairs = 6
    valid_lens = th.tensor([3, 2])
    
    x = th.ones([batch_size, num_queries, num_hiddens])  # torch.Size([2, 4, 100])
    y = th.ones([batch_size, num_kvpairs, num_hiddens])  # torch.Size([2, 6, 100])

    multi_head_attention = MultiHeadAttention(query_size=num_hiddens, key_size=num_hiddens, value_size=num_hiddens, 
                                              num_hiddens=num_hiddens, num_heads=num_heads, dropout=0.5, bias=False)
    multi_head_attention.eval()
    
    output_attn = multi_head_attention(queries=x, keys=y, values=y, valid_lens=valid_lens)
    
    
    

    