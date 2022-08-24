# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import networkx as nx
import numpy as np
from gensim.models.word2vec import Word2Vec



class DeepWalk(object):
    def __init__(self, config):
        self.config = config
        
    
    @staticmethod
    def _get_corpus(G, n, length):
        """
        方法: 构造语料库
        输入: G[networkx.classes.digraph.DiGraph], n[int], length[int]
        输出: lst_corpus[list]
        """
        lst_corpus = []
        
        for i in range(n):
            print(f"create seqs {i} start...")
            start_node = np.random.choice(G.nodes)
            lst_seqs = DeepWalk._get_seqs(G, start_node, length)
            lst_corpus.append(lst_seqs)
        
        return lst_corpus
    
    
    @staticmethod
    def _get_seqs(G, start_node, length):
        """
        方法: 构造单条语料
        输入: G[networkx.classes.digraph.DiGraph], start_node[int], length[int]
        输出: lst_seqs[list]
        """
        lst_seqs = [start_node]
        
        for j in range(length):
            current_node = lst_seqs[-1]
            lst_successor_nodes = list(G.successors(current_node))
            
            if lst_successor_nodes:
                next_node = np.random.choice(lst_successor_nodes)
                lst_seqs.append(next_node)
        
        return lst_seqs
    
    
    def fit(self, G):
        # 读取参数
        n = self.config.get("n")
        length = self.config.get("length")
        n_neg = self.config.get("n_neg")
        n_embedd = self.config.get("n_embedd")
        window_size = self.config.get("window_size")
        n_exposures = self.config.get("n_exposures")
        
        # 构造语料库
        lst_corpus = DeepWalk._get_corpus(G, n, length)
        print(f"lst_corpus\n{lst_corpus}")
        
        # Word2Vec
        model = Word2Vec(sentences=lst_corpus,
                         sg=1,  # 0-CBOW, 1-SkipGram
                         hs=0,  # 0-负采样，1-hierarchical softmax
                         negative=n_neg,
                         vector_size=n_embedd,
                         window=window_size,
                         min_count=n_exposures,  # 丢掉出现频次小于 n_exposures 的词
                         compute_loss=True,
                         seed=1)
        
        return model



if __name__ == "__main__":
    config = {"n_embedd": 10, "n": 10, "length": 30, "n_neg": 10, "window_size": 5, "n_exposures": 3}
    G = nx.fast_gnp_random_graph(n=30, p=0.5, directed=True)  # 快速随机生成一个有向图
    deep_walk = DeepWalk(config)
    model = deep_walk.fit(G)
    weights = model.wv.vectors
    print(weights.shape)
    
    model.wv.most_similar(0, topn=3)
    
    
