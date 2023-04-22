# -*- coding: utf-8 -*-
"""
https://arxiv.org/pdf/1711.05101.pdf
"""
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score)
import torch as th
from torch import nn
import torch.optim as optim


device = th.device("cuda" if th.cuda.is_available() else "cpu")
th.set_default_tensor_type(th.DoubleTensor)  # 64位

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/tools"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 构造数据集
x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                           shuffle=True, random_state=0)

b = np.ones_like(y).reshape(-1, 1)
x = np.concatenate([b, x], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, shuffle=True, 
                                                    stratify=y, random_state=0)

# ----------------------------------------------------------------------------------------------------------------
# config
threshold = 0.5
lr = 0.01
epochs = 100

objt = lambda y_true, y_hat: -1.0 * np.mean(y_true.reshape(-1, 1) * np.log(y_hat) + \
                                            (1-y_true).reshape(-1, 1) * np.log(1-y_hat))

# ----------------------------------------------------------------------------------------------------------------
# sklearn
model = LogisticRegression(max_iter=100, fit_intercept=False)
model.fit(x_train, y_train)
w = model.coef_

y_hat = model.predict_proba(x_test)
y_pred = np.where(y_hat[:,1] >= threshold, 1, 0)

confusion_matrix(y_true=y_test, y_pred=y_pred)
f1_score(y_true=y_test, y_pred=y_pred, pos_label=1)

# ----------------------------------------------------------------------------------------------------------------
# GD (Gradient Descent)
w = np.random.randn(x.shape[1]) * 0.001
w = w.reshape(-1, 1)

for t in range(epochs):
    y_hat = 1.0 / (1 + np.exp(-x_train.dot(w)))
    loss = objt(y_train, y_hat)
    print(f"epoch {t}  loss {loss:.4f}")
    dx = -1.0 * x_train.T.dot(y_train.reshape(-1, 1) - y_hat) / len(x_train)
    w = w - lr * dx
    
y_hat = 1.0 / (1 + np.exp(-x_test.dot(w)))
y_pred = np.where(y_hat >= threshold, 1, 0).reshape(-1)

confusion_matrix(y_true=y_test, y_pred=y_pred)
f1_score(y_true=y_test, y_pred=y_pred, pos_label=1)

# ----------------------------------------------------------------------------------------------------------------
# GDM (Gradient Descent with Momentum)
w = np.random.randn(x.shape[1]) * 0.001
w = w.reshape(-1, 1)
beta = 0.9
v = 0

def gradient_descent_with_momentum(beta, v, dx):
    v = beta * v + (1 - beta) * dx
    return v

for t in range(epochs):
    y_hat = 1.0 / (1 + np.exp(-x_train.dot(w)))
    loss = objt(y_train, y_hat)
    print(f"epoch {t}  loss {loss:.4f}")
    dx = -1.0 * x_train.T.dot(y_train.reshape(-1, 1) - y_hat) / len(x_train)
    v = gradient_descent_with_momentum(beta, v, dx)
    w = w - lr * v

y_hat = 1.0 / (1 + np.exp(-x_test.dot(w)))
y_pred = np.where(y_hat >= threshold, 1, 0).reshape(-1)

confusion_matrix(y_true=y_test, y_pred=y_pred)
f1_score(y_true=y_test, y_pred=y_pred, pos_label=1)

# ----------------------------------------------------------------------------------------------------------------
# Adam (Adaptive Moment Estimation)
w = np.random.randn(x.shape[1]) * 0.001
w = w.reshape(-1, 1)
beta1 = 0.9
beta2 = 0.999
v = 0
s = 0
decay = 0
eps = 10**-8

def adam(beta1, beta2, v, s, dx, t):
    v = beta1 * v + (1 - beta1) * dx
    s = beta2 * s + (1 - beta2) * (dx**2)
    v = v / (1 - beta1**t)
    s = s / (1 - beta2**t)
    return v, s

for t in range(1, epochs+1):
    y_hat = 1.0 / (1 + np.exp(-x_train.dot(w)))
    loss = objt(y_train, y_hat)
    print(f"epoch {t}  loss {loss:.4f}")
    dx = -1.0 * x_train.T.dot(y_train.reshape(-1, 1) - y_hat) / len(x_train)
    v, s = adam(beta1, beta2, v, s, dx, t)
    lrm = lr * (1 - decay)**t
    w = w - lrm * (v / (np.sqrt(s) + eps))

y_hat = 1.0 / (1 + np.exp(-x_test.dot(w)))
y_pred = np.where(y_hat >= threshold, 1, 0).reshape(-1)

confusion_matrix(y_true=y_test, y_pred=y_pred)
f1_score(y_true=y_test, y_pred=y_pred, pos_label=1)

# ----------------------------------------------------------------------------------------------------------------
# AdamW (Adam with decoupled weight decay)
w = np.random.randn(x.shape[1]) * 0.001
w = w.reshape(-1, 1)
beta1 = 0.9
beta2 = 0.999
v = 0
s = 0
decay = 0.01
eps = 10**-8
lamb = 1/8 * 0.001

for t in range(1, epochs+1):
    y_hat = 1.0 / (1 + np.exp(-x_train.dot(w)))
    loss = objt(y_train, y_hat)
    print(f"epoch {t}  loss {loss:.4f}")
    dx = -1.0 * x_train.T.dot(y_train.reshape(-1, 1) - y_hat) / len(x_train)
    v, s = adam(beta1, beta2, v, s, dx, t)
    lrm = lr * (1 - decay)**t
    w = w - lrm * (v / (np.sqrt(s) + eps) + lamb*w)

y_hat = 1.0 / (1 + np.exp(-x_test.dot(w)))
y_pred = np.where(y_hat >= threshold, 1, 0).reshape(-1)

confusion_matrix(y_true=y_test, y_pred=y_pred)
f1_score(y_true=y_test, y_pred=y_pred, pos_label=1)

# ----------------------------------------------------------------------------------------------------------------
# pyTorch
class Model(th.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.linear = th.nn.Linear(in_features=config.get("in_features"), 
                                   out_features=2, 
                                   bias=False)

    def forward(self, x):
        logits = self.linear(x)
        return logits

# 训练
config = {"in_features": x.shape[1]}

x_train = th.from_numpy(x_train)
y_train = th.LongTensor(y_train)
x_test = th.from_numpy(x_test)
y_test = th.LongTensor(y_test)

model = Model(config)
opti = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
# opti = optim.Adam(params=model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
# opti = optim.AdamW(params=model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
objt = nn.CrossEntropyLoss(reduction="mean")
epochs = 100

for t in range(epochs):
    model.train()
    logits = model(x_train)
    loss = objt(logits, y_train)
    print(f"epoch {t}  loss {loss:.4f}")
    
    opti.zero_grad()
    loss.backward()
    opti.step()

logits = model(x_test)
y_hat = th.softmax(logits, dim=1)
y_pred = th.argmax(y_hat, dim=1)

confusion_matrix(y_true=y_test, y_pred=y_pred)
f1_score(y_true=y_test, y_pred=y_pred, pos_label=1)

'''
Examples::
    >>> m = nn.Sigmoid()
    >>> loss = nn.BCELoss()
    >>> input = torch.randn(3, requires_grad=True)
    >>> target = torch.empty(3).random_(2)
    >>> output = loss(m(input), target)
    >>> output.backward()

Examples::
    >>> # Example of target with class indices
    >>> loss = nn.CrossEntropyLoss()
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.empty(3, dtype=torch.long).random_(5)
    >>> output = loss(input, target)
    >>> output.backward()
    >>>
    >>> # Example of target with class probabilities
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5).softmax(dim=1)
    >>> output = loss(input, target)
    >>> output.backward()
'''
