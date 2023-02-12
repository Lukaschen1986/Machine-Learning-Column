# -*- coding: utf-8 -*-
import os
import torch as th
from torch import nn
# from d2l import torch as d2l

print(th.__version__)
print(th.version.cuda)
print(th.backends.cudnn.version())
th.set_default_tensor_type(th.DoubleTensor)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
# BatchNorm vs LayerNorm
batch_size = 2
num_lens = 4
num_hiddens = 4
x = th.randn([batch_size, num_lens, num_hiddens], dtype=th.float64)

bn = nn.BatchNorm1d(num_features=num_hiddens)
ln = nn.LayerNorm(normalized_shape=[num_lens, num_hiddens])

x_bn = bn(x)  # 对所有的 batchs 和 hiddens，按照 lens 进行归一化
x_ln = ln(x)  # 对所有的 lens 和 hiddens，按照 batchs 进行归一化

# ----------------------------------------------------------------------------------------------------------------
# 残差连接后进行层规范化
class AddNorm(nn.Module):
    """
    self = AddNorm(norm_shape=num_hiddens, dropout=0.5)
    """
    def __init__(self, norm_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)
    
    
    def forward(self, x, y):
        output_norm = self.ln(self.dropout(y) + x)
        return output_norm



if __name__ == "__main__":
    x = th.randn([batch_size, num_lens, num_hiddens], dtype=th.float64)
    y = th.randn([batch_size, num_lens, num_hiddens], dtype=th.float64)
    
    add_norm = AddNorm(norm_shape=[num_lens, num_hiddens], dropout=0.5)
    add_norm.eval()
    
    output_norm = add_norm(x, y)  # 维度不变
    
