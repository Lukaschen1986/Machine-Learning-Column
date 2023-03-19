# -*- coding: utf-8 -*-
import os
import torch as th
from torch import nn
# from d2l import torch as d2l
from multi_head_attention import MultiHeadAttention
from add_norm import AddNorm
from position_wise_ffn import PositionWiseFFN

th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# Transformer编码器块
class EncoderBlock(nn.Module):
    """
    self = EncoderBlock(query_size, key_size, value_size, num_hiddens, num_heads,
                        norm_shape, ffn_num_input, ffn_num_hiddens, dropout)
    """
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads,
                 norm_shape, ffn_num_input, ffn_num_hiddens, dropout,
                 bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.multi_head_attention = MultiHeadAttention(query_size, key_size, value_size, 
                                                       num_hiddens, num_heads, dropout, bias)
        self.add_norm_1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm_2 = AddNorm(norm_shape, dropout)

    
    def forward(self, x, valid_lens):
        output_attn = self.multi_head_attention(queries=x, keys=x, values=x, valid_lens=valid_lens)
        output_norm_1 = self.add_norm_1(x, output_attn)
        output_ffn = self.ffn(output_norm_1)
        output_encoder = self.add_norm_2(output_norm_1, output_ffn)
        return output_encoder
        
        

if __name__ == "__main__":
    query_size = 24
    key_size = 24
    value_size = 24
    num_hiddens = 24
    num_heads = 8
    ffn_num_input = 24
    ffn_num_hiddens = 48
    dropout = 0.5
    
    batch_size = 2
    num_lens = 100
    norm_shape = [num_lens, num_hiddens]
    x = th.ones([batch_size, num_lens, num_hiddens])  # torch.Size([2, 100, 24])
    
    valid_lens = th.tensor([3, 2])
    encoder_blk = EncoderBlock(query_size, key_size, value_size, num_hiddens, num_heads,
                               norm_shape, ffn_num_input, ffn_num_hiddens, dropout)
    encoder_blk.eval()
    
    output_encoder = encoder_blk(x, valid_lens)
    output_encoder.shape  # torch.Size([2, 100, 24])


    
