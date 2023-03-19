# -*- coding: utf-8 -*-
"""
https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/bert.html
"""
import os
#import math
import torch as th
from torch import nn
#from d2l import torch as d2l

th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ---------------------------------------------------------------------------------------------------------------
# 下句预测模型
class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)
    
    
    def forward(self, x):
        return self.output(x)



if __name__ == "__main__":
    x = th.flatten(encoded_x, start_dim=1)  # torch.Size([2, 192])
    nsp = NextSentencePred(x.shape[-1])
    nsp_y_hat = nsp(x)  # torch.Size([2, 2])
    
    nsp_y_true = th.tensor([0, 1])
    objt = nn.CrossEntropyLoss(reduction="none")
    loss_nsp = objt(nsp_y_hat, nsp_y_true)
    
    