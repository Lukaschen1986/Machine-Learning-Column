# -*- coding: utf-8 -*-
import os
import math
import torch as th
from torch import nn
from d2l import torch as d2l
from position_encoding import PositionEncoding
from encoder_block import EncoderBlock

th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# Transformer编码器
class TransformerEncoder(d2l.Encoder):
    """
    self = TransformerEncoder(vocab_size, query_size, key_size, value_size, num_hiddens,
                              num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
                              dropout, bias=False)
    """
    def __init__(self, vocab_size, query_size, key_size, value_size, num_hiddens,
                 num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
                 dropout, bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, dropout)
        self.num_hiddens = num_hiddens
        self.encoder_blks = nn.Sequential()
        
        for i in range(num_layers):
            self.encoder_blks.add_module(name="block_" + str(i), 
                                         module=EncoderBlock(query_size, key_size, value_size, num_hiddens, num_heads,
                                                             norm_shape, ffn_num_input, ffn_num_hiddens, dropout,
                                                             bias))
    
    
    def forward(self, x, valid_lens, *args):
        x = self.embedding(x) * math.sqrt(self.num_hiddens)  # torch.Size([2, 100, 24])
        x = self.pos_encoding(x)  # torch.Size([2, 100, 24])
        self.encoder_attention_weights = [None] * len(self.encoder_blks)  # 用于可视化
        
        for (i, encoder_blk) in enumerate(self.encoder_blks):
            x = encoder_blk(x, valid_lens)
            self.encoder_attention_weights[i] = encoder_blk.multi_head_attention.attention.attention_weights
        
        return x
        
        

if __name__ == "__main__":
    vocab_size = 200
    num_layers = 2

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
    
    x = th.ones([batch_size, num_lens], dtype=th.long)  # torch.Size([2, 100])
    
    valid_lens = th.tensor([3, 2])
    encoder = TransformerEncoder(vocab_size, query_size, key_size, value_size, num_hiddens,
                                 num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
                                 dropout, bias=False)
    encoder.eval()
    
    output = encoder(x, valid_lens)
    output.shape  # torch.Size([2, 100, 24])
    
    

        