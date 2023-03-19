# -*- coding: utf-8 -*-
import os
import math
import torch as th
from torch import nn
from d2l import torch as d2l
from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder
from encoder_decoder import EncoderDecoder
# from position_encoding import PositionEncoding
# from encoder_block import EncoderBlock
# from decoder_block import DecoderBlock

print(th.__version__)
print(th.version.cuda)
print(th.backends.cudnn.version())
th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# 训练
num_hiddens = 24
num_layers = 2
dropout = 0.5
batch_size = 64
num_lens = 10
num_heads = 4
ffn_num_input = 24
ffn_num_hiddens = 48
query_size = 24
key_size = 24
value_size = 24
# norm_shape = [num_lens, num_hiddens]
norm_shape = [num_hiddens]

lr = 0.005
num_epochs = 200
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_lens)

encoder = TransformerEncoder(len(src_vocab), query_size, key_size, value_size, num_hiddens,
                             num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
                             dropout, bias=False)

decoder = TransformerDecoder(len(tgt_vocab), query_size, key_size, value_size, num_hiddens,
                             num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers,
                             dropout, bias=False)

net = EncoderDecoder(encoder, decoder)



if __name__ == "__main__":
    # https://zh-v2.d2l.ai/chapter_recurrent-modern/seq2seq.html
    
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    
    for (eng, fra) in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, 
                                                                    num_lens, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
        
        
        
        
        
        
        