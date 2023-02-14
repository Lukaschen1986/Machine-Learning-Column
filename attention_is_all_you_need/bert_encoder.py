# -*- coding: utf-8 -*-
import os
import math
import torch as th
from torch import nn
from d2l import torch as d2l
from encoder_block import EncoderBlock

print(th.__version__)
print(th.version.cuda)
print(th.backends.cudnn.version())
th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# 获取输入序列的词元及其片段索引


def get_token_and_segments(tokens_a, tokens_b=None):
    tokens = ["<cls>"] + tokens_a + ["<sep>"]
    segments = [0] * (len(tokens_a) + 2)

    if tokens_b is not None:
        tokens += tokens_b + ["<sep>"]
        segments += [1] * (len(tokens_b + 1))

    return tokens, segments


# ----------------------------------------------------------------------------------------------------------------
# Bert 编码器
class BertEncoder(nn.Module):
    def __init__(self, vocab_size, query_size, key_size, value_size, num_hiddens,
                 num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
                 dropout, max_len=1000, bias=False, **kwargs):
        super(BertEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.pos_encoding = nn.Parameter(th.randn(1, max_len, num_hiddens))
        self.encoder_blks = nn.Sequential()

        for i in range(num_layers):
            self.encoder_blks.add_module(name="block_" + str(i),
                                         module=EncoderBlock(query_size, key_size, value_size, num_hiddens, num_heads,
                                                             norm_shape, ffn_num_input, ffn_num_hiddens, dropout,
                                                             bias))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        x = self.token_embedding(tokens) + self.segment_embedding(segments)
        x = x + self.pos_encoding.data[:, 0:x.shape[1], :]

        for encoder_blk in self.encoder_blks:
            x = encoder_blk(x, valid_lens)

        return x