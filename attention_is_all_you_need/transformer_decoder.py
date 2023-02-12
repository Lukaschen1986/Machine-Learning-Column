# -*- coding: utf-8 -*-
import os
import math
import torch as th
from torch import nn
from d2l import torch as d2l
from position_encoding import PositionEncoding
from encoder_block import EncoderBlock
from decoder_block import DecoderBlock

print(th.__version__)
print(th.version.cuda)
print(th.backends.cudnn.version())
th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# Transformer解码器
class TransformerDecoder(d2l.AttentionDecoder):
    """
    self = TransformerDecoder(vocab_size, query_size, key_size, value_size, num_hiddens,
                              num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
                              dropout, bias=False)
    """
    def __init__(self, vocab_size, query_size, key_size, value_size, num_hiddens,
                 num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
                 dropout, bias=False, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, dropout)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.decoder_blks = nn.Sequential()
        
        for i in range(num_layers):
            self.decoder_blks.add_module(name="block_" + str(i), 
                                         module=DecoderBlock(query_size, key_size, value_size, num_hiddens, num_heads,
                                                             norm_shape, ffn_num_input, ffn_num_hiddens, dropout, i, 
                                                             bias=False))
        
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    
    # def init_state(self, output_encoder, valid_lens_encoder, *args):
    #     return [output_encoder, valid_lens_encoder, [None] * self.num_layers]
    
    def init_state(self, output_encoder, valid_lens_encoder, *args):
        self.seq_x = None
        return [output_encoder, valid_lens_encoder]
    
    
    def forward(self, x, state):
        x = self.embedding(x) * math.sqrt(self.num_hiddens)  # torch.Size([2, 100, 24])
        x = self.pos_encoding(x)  # torch.Size([2, 100, 24])
        # self.decoder_attention_weights = [[None] * len(self.decoder_blks) for _ in range(2)]  # 用于可视化
        self._attention_weights = [[None] * len(self.decoder_blks) for _ in range(2)]  # 用于可视化
        
        if not self.training:
            self.seq_x = x if self.seq_x is None else th.cat((self.seq_x, x), dim=1)
            x = self.seq_x
        
        for (i, decoder_blk) in enumerate(self.decoder_blks):
            x, state = decoder_blk(x, state)
            
            # 解码器 自注意力权重
            # self.decoder_attention_weights[0][i] = decoder_blk.multi_head_attention_1.attention.attention_weights
            self._attention_weights[0][i] = decoder_blk.multi_head_attention_1.attention.attention_weights
            
            # 编码器－解码器 自注意力权重
            # self.decoder_attention_weights[1][i] = decoder_blk.multi_head_attention_2.attention.attention_weights
            self._attention_weights[1][i] = decoder_blk.multi_head_attention_2.attention.attention_weights
        
        # 输出层全连接
        if not self.training:
            return self.dense(x)[:, -1:, :], state
        else:
            return self.dense(x), state
    
    
    @property
    def attention_weights(self):
        return self._attention_weights
    
    
    # def forward(self, x, state):
    #     x = self.embedding(x) * math.sqrt(self.num_hiddens)  # torch.Size([2, 100, 24])
    #     x = self.pos_encoding(x)  # torch.Size([2, 100, 24])
    #     self.decoder_attention_weights = [[None] * len(self.decoder_blks) for _ in range(2)]  # 用于可视化
        
    #     for (i, decoder_blk) in enumerate(self.decoder_blks):
    #         x, state = decoder_blk(x, state)
            
    #         # 解码器 自注意力权重
    #         self.decoder_attention_weights[0][i] = decoder_blk.multi_head_attention_1.attention.attention_weights
            
    #         # 编码器－解码器 自注意力权重
    #         self.decoder_attention_weights[1][i] = decoder_blk.multi_head_attention_2.attention.attention_weights
        
    #     # 输出层全连接
    #     x = self.dense(x)
    #     return x, state
    
    

# if __name__ == "__main__":
#     vocab_size = 200
#     num_layers = 2

#     query_size = 24
#     key_size = 24
#     value_size = 24
#     num_hiddens = 24
#     num_heads = 8
#     ffn_num_input = 24
#     ffn_num_hiddens = 48
#     dropout = 0.5
    
#     batch_size = 2
#     num_lens = 100
#     norm_shape = [num_lens, num_hiddens]
    
#     x = th.ones([batch_size, num_lens, num_hiddens])
    
#     valid_lens = th.tensor([3, 2])
#     encoder_blk = EncoderBlock(query_size, key_size, value_size, num_hiddens, num_heads,
#                                norm_shape, ffn_num_input, ffn_num_hiddens, dropout)
#     state = [encoder_blk(x, valid_lens), valid_lens, [None]]
    
#     decoder = TransformerDecoder(vocab_size, query_size, key_size, value_size, num_hiddens,
#                                  num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
#                                  dropout, bias=False)
#     decoder.eval()
    
#     output = decoder(x, state)
#     output.shape  # torch.Size([2, 100, 24])
    
    
    
    
    