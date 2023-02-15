# -*- coding: utf-8 -*-
import os
import torch as th
from torch import nn
# from d2l import torch as d2l
from multi_head_attention import MultiHeadAttention
from add_norm import AddNorm
from position_wise_ffn import PositionWiseFFN
from encoder_block import EncoderBlock

th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# Transformer解码器
class DecoderBlock(nn.Module):
    """
    self = DecoderBlock(query_size, key_size, value_size, num_hiddens, num_heads,
                        norm_shape, ffn_num_input, ffn_num_hiddens, dropout, i, 
                        bias=False)
    """
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads,
                 norm_shape, ffn_num_input, ffn_num_hiddens, dropout, i, 
                 bias=False, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        # 第一个注意力块
        self.multi_head_attention_1 = MultiHeadAttention(query_size, key_size, value_size, 
                                                         num_hiddens, num_heads, dropout, bias)
        self.add_norm_1 = AddNorm(norm_shape, dropout)
        # 第二个注意力块
        self.multi_head_attention_2 = MultiHeadAttention(query_size, key_size, value_size, 
                                                         num_hiddens, num_heads, dropout, bias)
        self.add_norm_2 = AddNorm(norm_shape, dropout)
        # 前馈
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        # 第三个规范化块
        self.add_norm_3 = AddNorm(norm_shape, dropout)
        
    
    def forward(self, x, state):
        output_encoder, valid_lens_encoder = state[0], state[1]
        batch_size, num_lens, _ = x.shape
        valid_lens_decoder = th.arange(1, num_lens + 1, device=x.device).repeat(batch_size, 1)
        
        # 自注意力
        output_attn_1 = self.multi_head_attention_1(queries=x, keys=x, values=x, 
                                                    valid_lens=valid_lens_decoder)
        output_norm_1 = self.add_norm_1(x, output_attn_1)
        
        # 编码器－解码器注意力
        output_attn_2 = self.multi_head_attention_2(queries=output_norm_1, keys=output_encoder, 
                                                    values=output_encoder, valid_lens=valid_lens_encoder)
        output_norm_2 = self.add_norm_2(output_norm_1, output_attn_2)
        
        # 前馈
        output_ffn = self.ffn(output_norm_2)  # torch.Size([2, 100, 24])
        
        # 第三个规范化块
        output_decoder = self.add_norm_3(output_norm_2, output_ffn)  # torch.Size([2, 100, 24])
        
        return output_decoder, state
    
    
    # def forward(self, x, state):
    #     """
    #     训练阶段，输出序列的所有词元都在同一时间处理，
    #     因此state[2][self.i]初始化为None。
    #     预测阶段，输出序列是通过词元一个接着一个解码的，
    #     因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
    #     """
    #     output_encoder, valid_lens_encoder = state[0], state[1]
        
    #     if state[2][self.i] is None:
    #         # 训练阶段，key 和 value 就是 x 本身
    #         key_values = x  # torch.Size([2, 100, 24])
    #     else:
    #         # 推理阶段，key 和 value 是前面训练好的累计值的 concat
    #         key_values = th.cat([state[2][self.i], x], axis=1)
        
    #     # 把当前的 key_values 存入 state[2][self.i]
    #     state[2][self.i] = key_values
        
    #     if self.training:
    #         batch_size, num_lens, _ = x.shape
    #         valid_lens_decoder = th.arange(1, num_lens + 1, device=x.device).repeat(batch_size, 1)
    #     else:
    #         valid_lens_decoder = None
        
    #     # 自注意力
    #     output_attn_1 = self.multi_head_attention_1(queries=x, keys=key_values, values=key_values, 
    #                                                 valid_lens=valid_lens_decoder)
    #     output_norm_1 = self.add_norm_1(x, output_attn_1)
        
    #     # 编码器－解码器注意力
    #     output_attn_2 = self.multi_head_attention_2(queries=output_norm_1, keys=output_encoder, 
    #                                                 values=output_encoder, valid_lens=valid_lens_encoder)
    #     output_norm_2 = self.add_norm_2(output_norm_1, output_attn_2)
        
    #     # 前馈
    #     output_ffn = self.ffn(output_norm_2)  # torch.Size([2, 100, 24])
        
    #     # 第三个规范化块
    #     output_decoder = self.add_norm_3(output_norm_2, output_ffn)  # torch.Size([2, 100, 24])
        
    #     return output_decoder, state
        
        

if __name__ == "__main__":
    i = 0
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
    
    # state = [encoder_blk(x, valid_lens), valid_lens, [None]]
    state = [encoder_blk(x, valid_lens), valid_lens]
    
    decoder_blk = DecoderBlock(query_size, key_size, value_size, num_hiddens, num_heads,
                               norm_shape, ffn_num_input, ffn_num_hiddens, dropout, i)
    output_decoder, state = decoder_blk(x, state)
    output_decoder.shape  # torch.Size([2, 100, 24])
        
    
                 