# -*- coding: utf-8 -*-
"""
https://www.bilibili.com/video/BV1v3411r78R?p=1&vd_source=fac9279bd4e33309b405d472b24286a8
"""
import os
import sys
import numpy as np
import pandas as pd
import torch as th
from torch import nn

print("PyTorch version:", th.__version__)
print("use gpu:", th.cuda.is_available())
device = th.device("cuda" if th.cuda.is_available() else "cpu")

path_project = "C:/my_project/MyGit/Machine-Learning-Column/"

# ---------------------------------------------------------------------------------------------------------------
# 输入向量
d = 3
a1 = th.randn([d, 1])  # torch.Size([3, 1])
a2 = th.randn([d, 1])
a3 = th.randn([d, 1])
a4 = th.randn([d, 1])

# ---------------------------------------------------------------------------------------------------------------
# 定义 Wq, Wk, Wv
Wq = th.randn([d, d], requires_grad=True)
Wk = th.randn([d, d], requires_grad=True)
Wv = th.randn([d, d], requires_grad=True)

# ---------------------------------------------------------------------------------------------------------------
# 计算 q, k, v
q1 = th.matmul(Wq, a1)

k1 = th.matmul(Wk, a1)
k2 = th.matmul(Wk, a2)
k3 = th.matmul(Wk, a3)
k4 = th.matmul(Wk, a4)

a1_1 = th.matmul(q1.T, k1)
a1_2 = th.matmul(q1.T, k2)
a1_3 = th.matmul(q1.T, k3)
a1_4 = th.matmul(q1.T, k4)
a1_1, a1_2, a1_3, a1_4 = th.softmax(th.cat([a1_1, a1_2, a1_3, a1_4], dim=1) / np.sqrt(d), dim=1)[0]

v1 = th.matmul(Wv, a1)
v2 = th.matmul(Wv, a2)
v3 = th.matmul(Wv, a3)
v4 = th.matmul(Wv, a4)
b1 = a1_1*v1 + a1_2*v2 + a1_3*v3 + a1_4*v4

# ---------------------------------------------------------------------------------------------------------------
# 矩阵形式
I = th.cat([a1, a2, a3, a4], dim=1)  # torch.Size([3, 4])
Q = th.matmul(Wq, I)
K = th.matmul(Wk, I)
V = th.matmul(Wv, I)

A = th.matmul(Q.T, K)
A_ = th.softmax(A / np.sqrt(d), dim=1)
O = th.matmul(V, A_.T)  # torch.Size([3, 4])，每一列为一个b向量


