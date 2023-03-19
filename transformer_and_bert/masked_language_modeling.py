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
# 掩码语言模型
class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
                nn.Linear(num_inputs, num_hiddens),
                nn.ReLU(),
                nn.LayerNorm(num_hiddens),
                nn.Linear(num_hiddens, vocab_size)
                )
        
    
    def forward(self, x, pred_positions):
        # 取出需要预测的 mask 张量
        num_pred_positions = pred_positions.shape[1]  # 3
        pred_positions = pred_positions.reshape(-1)  # tensor([1, 5, 2, 6, 1, 5])
        
        batch_size = x.shape[0]  # 2
        batch_idx = th.arange(0, batch_size)
        batch_idx = th.repeat_interleave(input=batch_idx, repeats=num_pred_positions, dim=0)  # slzb
        
        masked_x = x[batch_idx, pred_positions]  # torch.Size([6, 24])
        masked_x = masked_x.reshape([batch_size, num_pred_positions, -1])  # torch.Size([2, 3, 24])
        
        # 送入 mlp，预测每个 mask 在 vocab_size 上的概率
        mlm_y_hat = self.mlp(masked_x)  # torch.Size([2, 3, 200])
        return mlm_y_hat



if __name__ == "__main__":
    vocab_size = 200
    num_hiddens = 24
    num_inputs = ffn_num_input = 24
    mlm = MaskLM(vocab_size, num_hiddens, num_inputs)
    
    mlm_y_true = th.tensor([
            [7, 8, 9],
            [10, 20, 30]
            ])
    
    x = encoded_x  # 来自于 BertEncoder 输出  torch.Size([2, 8, 24])
    pred_positions = th.tensor([
            [1, 5, 2],
            [6, 1, 5]
            ])
    mlm_y_hat = mlm(x, pred_positions)
    mlm_y_hat.shape  # torch.Size([2, 3, 200])
    
    objt = nn.CrossEntropyLoss(reduction="none")
    loss_mlm = objt(mlm_y_hat.reshape([-1, vocab_size]), mlm_y_true.reshape(-1))
    
    
