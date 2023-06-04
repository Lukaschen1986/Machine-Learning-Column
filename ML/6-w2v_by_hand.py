# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
from collections import Counter
import numpy as np
import torch as tc
import torch.nn as nn
import torch.optim as optim
from pytorchtools import EarlyStopping


print(tc.__version__)
print(tc.version.cuda)
print(tc.backends.cudnn.version())
tc.manual_seed(1)
tc.set_default_tensor_type(tc.DoubleTensor)
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")


# 定义参数
n_embedd = 8
batch_size = 16
window_size = 5
n_neg = 10


# 构造语料库
lst_corpus = [...]


# 构造语料字典
set_vocab = set(lst_corpus)
word2idx = {word: idx for (idx, word) in enumerate(set_vocab)}
idx2word = {idx: word for (idx, word) in enumerate(set_vocab)}


# 以索引的形式构造语料库
lst_corpus_idx = {word2idx.get(word) for word in lst_corpus}


# 计算占比
counter = Counter(lst_corpus_idx)
dct_freqs = {idx: count/len(lst_corpus_idx) for (idx, count) in counter.items()}
dct_freqs = dict(sorted(dct_freqs.items(), key=lambda x: x[0]))


# 负采样分布
probs = np.array(list(dct_freqs.values()))
neg_distri = tc.from_numpy(probs**0.75 / np.sum(probs**0.75))


# 定义 mini-batch 方法
def get_context(batch, j, window_size):
    """
    方法: 根据中心词，获取上下文词
    输入: batch[list], j[int], window_size[int]
    输出: lst_context[list]
    """
    dynamic_window_size = np.random.randint(1, window_size + 1)  # 动态窗口
    start = ((j - dynamic_window_size) if (j - dynamic_window_size) >= 0 else 0)
    end = j + dynamic_window_size
    
    set_context = set(batch[start: j] + batch[j+1: end+1])  # 上下文去重
    lst_context = list(set_context)
    return lst_context


def get_batch(lst_corpus_idx, batch_size, window_size):
    """
    方法: yield mini-batch 方法
    输入: lst_corpus_idx[list], batch_size[int], window_size[int]
    输出: lst_center[list], lst_context[list]
    """
    for i in range(0, len(lst_corpus_idx), batch_size):
        batch = lst_corpus_idx[i: i+batch_size]
        lst_center, lst_context = [], []
        
        for j in range(len(batch)):
            x_center = batch[j]
            x_context = get_context(batch, j, window_size)
            lst_center.extend([x_center]*len(x_context))  # 保持 lst_center 和 lst_context 样本量一致
            lst_context.extend(x_context)
        
        yield lst_center, lst_context


# 定义网络结构
class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embedd, neg_distri):
        super(SkipGram, self).__init__()
        self.n_vocab = n_vocab
        self.n_embedd = n_embedd
        self.neg_distri = neg_distri
    
        # embeddding layer
        self.center_embedding = nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.n_embedd)
        self.context_embedding = nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.n_embedd)
        
        # init weight
        self.center_embedding.weight.data.normal_(0, 0.01)
        self.context_embedding.weight.data.normal_(0, 0.01)
    
    
    def f_center(self, tc_center):
        v = self.center_embedding(tc_center)
        return v
    
    
    def f_context(self, tc_context):
        u = self.context_embedding(tc_context)
        return u
    
    
    def f_neg_samples(self, n_pos, n_neg):
        tc_context_neg = tc.multinomial(input=self.neg_distri,
                                        num_samples=n_pos*n_neg,
                                        replacement=True)  # slzb
        tc_context_neg = tc_context_neg.to(device)
        u_neg = self.context_embedding(tc_context_neg).reshape(n_pos, n_neg, self.n_embedd)
        return u_neg


# 定义损失函数
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    
    def forward(self, v, u, u_neg):
        N, p = v.shape
        v_update = v.reshape(N, p, 1)
        u_update = u.reshape(N, 1, p)
        
        # 正样本损失
        loss_pos = tc.bmm(u_update, v_update).sigmod().log()
        loss_pos = loss_pos.squeeze()  # 维度压缩
        
        # 负样本损失
        loss_neg = tc.bmm(u_neg.neg(), v_update).sigmod().log()  # .neg() 取负号
        loss_neg = loss_neg.squeeze().sum(dim=1)  # 维度压缩，再按行求和
        
        # 合并损失
        return -(loss_pos + loss_neg).mean()


# model
n_vocab = len(set_vocab)
estimator = SkipGram(n_vocab, n_embedd, neg_distri).to(device)
objt = Loss()
opti = optim.Adam(params=estimator.parameters(), lr=0.003, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
epochs = 1000
patience = 10
early_stopping = EarlyStopping(patience, verbose=False)
evals_res = {"train": {"loss": []}, "valid": {"loss": []}}


# 训练
for epoch in range(epochs):
    loss_train = 0
    batchs = 0
    estimator.train()
    for (lst_center, lst_context) in get_batch(lst_corpus_idx, batch_size, window_size):
        # batch
        tc_center = tc.LongTensor(lst_center).to(device)
        tc_context = tc.LongTensor(lst_context).to(device)
        
        # forward
        v = estimator.f_center(tc_center)
        u = estimator.f_context(tc_context)
        n_pos = v.shape[0]
        u_neg = estimator.f_neg_samples(n_pos, n_neg)
        
        # loss
        loss = objt(v, u, u_neg)
        loss_train += loss
        batchs += 1
        
        # backward
        opti.zero_grad()
        loss.backward()
        opti.step()
    
    loss_train = float(loss_train.cpu().detach().numpy())
    loss_train /= batchs
    evals_res["train"]["loss"].append(loss_train)
    
    early_stopping(loss_train, estimator)
    if early_stopping.early_stop:
        break
    
    # log
    print(f"epoch {epoch}  train-loss {loss_train:.4f}")


# 权重
weights = estimator.state_dict()["center_embedding.weight"].cpu()

