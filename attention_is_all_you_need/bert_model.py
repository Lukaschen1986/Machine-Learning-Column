# -*- coding: utf-8 -*-
"""
https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/bert.html
"""
import os
#import math
import torch as th
from torch import nn
#from d2l import torch as d2l
from bert_encoder import BertEncoder
from masked_language_modeling import MaskLM
from next_sentence_prediction import NextSentencePred

th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ---------------------------------------------------------------------------------------------------------------
# 整合 Bert 模型
class BertModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super(BertModel, self).__init__()
        
        self.encoder = BertEncoder(vocab_size, query_size, key_size, value_size, num_hiddens,
                                   num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
                                   dropout)
        
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        
        self.hidden = nn.Sequential(
                nn.Linear(hid_in_features, num_hiddens),
                nn.Tanh()
                )
        self.nsp = NextSentencePred(nsp_in_features)
        
    
    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_x = self.encoder(tokens, segments, valid_lens)
        
        if pred_positions is not None:
            mlm_y_hat = self.mlm(encoded_x, pred_positions)
        else:
            mlm_y_hat = None
        
        # 用于 下句预测模型 的多层感知机分类器的隐藏层，0是 <cls> 标记的索引
        nsp_y_hat = self.nsp(self.hidden(encoded_x[:, 0, :]))        
        return encoded_x, mlm_y_hat, nsp_y_hat
        


if __name__ == "__main__":
    vocab_size = 200
    num_hiddens = 768
    norm_shape = [num_hiddens]
    ffn_num_input = 768
    ffn_num_hiddens = 768
    num_heads = 4
    num_layers = 2
    dropout = 0.5
    max_len = 1000
    key_size = 768
    query_size = 768
    value_size = 768
    hid_in_features = 768
    mlm_in_features = 768
    nsp_in_features = 768
    
    tokens = th.randint(0, vocab_size, (2, 8))
    segments = th.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])

    bert = BertModel(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                     ffn_num_hiddens, num_heads, num_layers, dropout,
                     max_len, key_size, query_size, value_size,
                     hid_in_features, mlm_in_features, nsp_in_features)
    bert.eval()
    
    encoded_x, mlm_y_hat, nsp_y_hat = bert(tokens, segments)
    
    
    
    