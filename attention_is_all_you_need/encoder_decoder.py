# -*- coding: utf-8 -*-
import os
import torch as th
from torch import nn

th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# 编码器-解码器架构的基类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, encoder_x, decoder_x, *args):
        output_encoder = self.encoder(encoder_x, *args)
        state = self.decoder.init_state(output_encoder, *args)
        output_decoder = self.decoder(decoder_x, state)
        return output_decoder