# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import networkx as nx
import numpy as np
from gensim.models.word2vec import Word2Vec



class Node2Vec(object):
    def __init__(self, config):
        self.config = config
    
    
    @staticmethod
    def _get_corpus(G, n, length, p, q):
        """
        方法: 构造语料库
        输入: G[networkx.classes.digraph.DiGraph], n[int], length[int], p[float], q[float]
        输出: lst_corpus[list]
        """
        lst_corpus = []
        
        for i in range(n):
            print(f"create seqs {i} start...")
            start_node = np.random.choice(G.nodes)
            lst_seqs = Node2Vec._get_seqs(G, start_node, length, p, q)
            lst_corpus.append(lst_seqs)
        
        return lst_corpus
    
    
    @staticmethod
    def _get_seqs(G, start_node, length, p, q):
        """
        方法: 构造单条语料
        输入: G[networkx.classes.digraph.DiGraph], start_node[int], length[int], p[float], q[float]
        输出: lst_seqs[list]
        """
        lst_seqs = [None, start_node]
        
        for j in range(length):
            v = lst_seqs[-1]
            t = lst_seqs[-2]
            lst_x, arr_probs = Node2Vec._get_trans_info(G, v, t, p, q)
            
            if lst_x:
                x = np.random.choice(lst_x, p=arr_probs)
                lst_seqs.append(x)
        
        return lst_seqs[1: ]
    
    
    @staticmethod
    def _get_trans_info(G, v, t, p, q):
        """
        方法: 计算后继结点与转移矩阵（node2vec 核心步骤）
        输入: G[networkx.classes.digraph.DiGraph], v[int], t[int], p[float], q[float]
        输出: lst_x[list], arr_probs[numpy.ndarray]
        """
        lst_x = list(G.successors(v))
        Z = len(lst_x)
        
        arr_w = Node2Vec._get_weight(G, v, lst_x)
        arr_a = Node2Vec._get_alpha(G, t, lst_x, p, q)
        
        arr_pi = arr_a * arr_w
        arr_probs = arr_pi / Z
        arr_probs /= np.sum(arr_probs)
        return lst_x, arr_probs
    
    
    @staticmethod
    def _get_weight(G, v, lst_x):
        """
        方法: 计算后继结点权重
        输入: G[networkx.classes.digraph.DiGraph], v[int], lst_x[list]
        输出: arr_w[numpy.ndarray]
        """
        arr_w = np.array([])
        
        for x in lst_x:
            w = G.edges[v, x].get("weight")
            arr_w = np.append(arr_w, w)
        
        return arr_w
    
    
    @staticmethod
    def _get_alpha(G, t, lst_x, p, q):
        """
        方法: 计算 alpha
        输入: G[networkx.classes.digraph.DiGraph], t[int], lst_x[list], p[float], q[float]
        输出: arr_a[numpy.ndarray]
        """
        if not t:
            arr_a = np.ones(len(lst_x))
        else:
            arr_a = np.array([])
            lst_v = list(G.successors(t))
            
            for x in lst_x:
                if x == t:
                    a = 1 / p
                elif x in lst_v:
                    a = 1
                else:
                    a = 1 / q
                arr_a = np.append(arr_a, a)
        
        return arr_a
    
    
    def fit(self, G):
        # 读取参数
        n = self.config.get("n")
        length = self.config.get("length")
        n_neg = self.config.get("n_neg")
        n_embedd = self.config.get("n_embedd")
        window_size = self.config.get("window_size")
        n_exposures = self.config.get("n_exposures")
        p = self.config.get("p")
        q = self.config.get("q")
        
        # 构造语料库
        lst_corpus = Node2Vec._get_corpus(G, n, length, p, q)
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
    config = {"n_embedd": 10, "n": 10, "length": 30, "n_neg": 10, "window_size": 5, "n_exposures": 3, 
              "p": 2, "q": 5}
    
    # 快速随机生成一个有向图
    G = nx.fast_gnp_random_graph(n=10, p=0.5, directed=True)
    
    # 添加边权重
    for (start, ends) in G.adj.items():
        for (end, _) in ends.items():
            G.add_weighted_edges_from([
                    (start, end, np.round(np.random.uniform(), 4))
                    ])
    G.adj
    
    # save and load
    nx.write_gml(G, path)
    G = nx.read_gml(path)
    
    # model
    node2vec = Node2Vec(config)
    model = node2vec.fit(G)
    weights = model.wv.vectors
    print(weights.shape)
    
    model.wv.most_similar(0, topn=3)
    '''
    [
    (5, 0.8938447833061218), 
    (4, 0.8735263347625732), 
    (3, 0.8707155585289001)
    ]
    '''
    
