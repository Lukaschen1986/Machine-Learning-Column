# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import os
import sys
import pickle
import torch as tc
from torchtext.legacy.data import (Dataset, Field, Example, BucketIterator)
from tqdm import tqdm
from torchcrf import CRF
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


print(tc.__version__)
print(tc.version.cuda)
print(tc.backends.cudnn.version())
tc.manual_seed(1)
tc.set_default_tensor_type(tc.DoubleTensor)


# 定义数据结构
class MyDataset(Dataset):
    def __init__(self, lst_bio, field_word, field_tag, predict=False):
        fields_info = [("word", field_word), ("tag", field_tag)]
        lst = []
        
        if predict:
            for (lst_words, _) in tqdm(lst_bio):
                example = Example.fromlist(data=[lst_words, []], fields=fields_info)
                lst.append(example)
        else:
            for (lst_words, lst_tags) in tqdm(lst_bio):
                example = Example.fromlist(data=[lst_words, lst_tags], fields=fields_info)
                lst.append(example)
        
        Dataset.__init__(self, lst, fields_info)


# 定义网络结构
class BiLSTM_CRF(nn.Module):
    def __init__(self, config, use_pretrain=True):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = config.get("vocab_size")
        self.tagset_size = config.get("tagset_size")
        self.embedding_dim = config.get("embedding_dim")
        self.hidden_dim = config.get("hidden_dim")
        
        # init layers
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=self.embedding_dim)
        if use_pretrain:
            pretrain_weights = config.get("pretrain")
            self.embedding_layer.weight = nn.Parameter(tc.from_numpy(pretrain_weights))
        else:
            self.embedding_layer.weight.data.normal_(0, 0.01)
        
        self.lstm_layer = nn.LSTM(input_size=self.embedding_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=1,
                                  dropout=config.get("dropout"),
                                  batch_first=True,
                                  bidirectional=True)
        self.fc_layer = nn.Linear(in_features=self.hidden_dim*2,
                                  out_features=self.tagset_size)
        self.crf = CRF(num_tags=self.tagset_size, batch_first=True)
        
    
    def f(self, tnr_word_idxs):
        # embedding layer
        embeds = self.embedding_layer(tnr_word_idxs)
        
        # lstm layer
        h0 = tc.randn([2, embeds.shape[0], self.hidden_dim]) * 0.01
        c0 = tc.randn([2, embeds.shape[0], self.hidden_dim]) * 0.01
        lstm_out, [ht, ct] = self.lstm_layer(embeds, [h0, c0])
        
        # fc layer
        lstm_feats = self.fc_layer(lstm_out)
        return lstm_feats
    
    
    def objt(self, lstm_feats, tnr_tag_idxs):
        likelihood = self.crf(lstm_feats, tnr_tag_idxs)
        loss = -1.0 * likelihood
        return loss
    
    
    def decode(self, tnr_word_idxs):
        lstm_feats = self.f(tnr_word_idxs)
        lst_tag_idxs = self.crf.decode(lstm_feats)
        return lst_tag_idxs
    


if __name__ == "__main__":
    # 构造数据集
    lst_bio = [
            (["a", "p", "p"], ["B", "I", "I"]),
            ]
    
    field_word = Field(sequential=True, use_vocab=True, init_token="<start>", eos_token="<stop>")
    field_tag = Field(sequential=True, use_vocab=True, init_token="<start>", eos_token="<stop>")
    
    BATCH_SIZE = 128
    main_train = MyDataset(lst_bio, field_word, field_tag)
    field_word.build_vocab(main_train)
    field_tag.build_vocab(main_train)
    
    iter_train = BucketIterator(dataset=main_train,
                                batch_size=BATCH_SIZE,
                                sort_key=lambda x: len(x.word),
                                shuffle=True,
                                sort_within_batch=False,
                                repeat=False)
    
    # 模型训练
    config = {...}
    model = BiLSTM_CRF(config, use_pretrain=True)
    opti = optim.Adam(params=model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=10**-8, weight_decay=0.01)
    lst_loss_train = []
    
    for epoch in range(100):
        # train
        loss_tmp = 0
        model.train()
        for idx, ((tnr_word_idxs, tnr_tag_idxs), _) in enumerate(iter_train):
            tnr_word_idxs = tnr_word_idxs.permute(1, 0)
            tnr_tag_idxs = tnr_tag_idxs.permute(1, 0)
            
            lstm_feats = model.f(tnr_word_idxs)
            loss = model.objt(lstm_feats, tnr_tag_idxs)
            loss_tmp += loss.item()
            
            opti.zero_grad()
            loss.backward()
            opti.step()
        
        loss_tmp = loss_tmp / BATCH_SIZE
        lst_loss_train.append(loss_tmp)
        print(f"epoch {epoch} loss_train {loss_tmp:.4f}")
    
    plt.plot(lst_loss_train)


