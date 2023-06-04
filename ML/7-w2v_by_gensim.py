# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import os
import torch as tc
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary


# 构造语料库
lst_corpus = [
        [...],
        [...]
        ]


# 定义参数
cpus = os.cpu_count()
n_embedd = 8
window_size = 5
n_exposures = 2
n_iterations = 5
n_neg = 10


# model
estimator = Word2Vec(sentences=lst_corpus,
                     sg=1,  # 0-CBOW, 1-SkipGram
                     hs=0, # 0-负采样，1-hierarchical softmax
                     negative=n_neg,
                     vector_size=n_embedd,
                     window=window_size,
                     min_count=n_exposures,  # 丢掉出现频次小于 n_exposures 的词
                     workers=cpus,
                     epochs=n_iterations,
                     compute_loss=True,
                     seed=1)

gensim_dict = Dictionary()
gensim_dict.doc2bow(document=estimator.wv.vocab.keys(),
                    allow_update=True,
                    return_missing=False)

word2idx = {word: idx for (idx, word) in gensim_dict.items()}
idx2word = {word: estimator.get(word) for (word, _) in word2idx.items()}
