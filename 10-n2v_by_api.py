# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import os
import networkx as nx
import numpy as np
from node2vec import Node2Vec



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
    
    node2vec = Node2Vec(G, 
                        dimensions=config.get("n_embedd"), 
                        num_walks=config.get("n"), 
                        walk_length=config.get("length"), 
                        p=config.get("p"), 
                        q=config.get("q"), 
                        workers=os.cpu_count())
    
    model = node2vec.fit()
    weights = model.wv.vectors
    print(weights.shape)
    model.wv.most_similar("0", topn=3)
    '''
    [('8', 0.9910511374473572),
     ('7', 0.9844254851341248),
     ('6', 0.9809565544128418)]
    '''
    
