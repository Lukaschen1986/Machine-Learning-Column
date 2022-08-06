# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 20:07:38 2022

@author: lukas

https://www.osgeo.cn/networkx/tutorial.html#attributes
"""

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot

# ----------------------------------------------------------------------------------------------------------------
# 无向图
G = nx.Graph(day="Friday")
G.graph  # {'day': 'Friday'}
G.graph["day"] = "Monday"
G.graph  # {'day': 'Monday'}

# ----------------------------------------------------------------------------------------------------------------
# 结点与属性
G.add_node(1, time="5pm")
G.add_nodes_from([2, 3])
G.add_nodes_from([3], time="2pm")
G.add_nodes_from([
        (4, {"color": "red"}),
        (5, {"color": "green"})
        ])

G.nodes  # NodeView((1, 2, 3, 4, 5))
list(G.nodes)  # [1, 2, 3, 4, 5]
G.nodes.data()
'''
NodeDataView(
        {1: {'time': '5pm', 'room': 714}, 
        2: {}, 
        3: {'time': '2pm'}, 
        4: {'color': 'red'}, 
        5: {'color': 'green'}}
        )
'''
G.nodes[1]  # {'time': '5pm'}
G.nodes[2]  # {}
G.nodes[3]  # {'time': '2pm'}
G.nodes[4]  # {'color': 'red'}
G.nodes[1]["room"] = 714  # 修改/增加结点的属性
G.nodes[1]  # {'time': '5pm', 'room': 714}

G.add_node("spam")  # adds node "spam"
G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'

# ----------------------------------------------------------------------------------------------------------------
# 一个图中的节点可以合并到另一个图中
H = nx.path_graph(10)
H.nodes  # NodeView((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_nodes_from(H)
G.nodes.data()
'''
NodeDataView({
        1: {'time': '5pm', 'room': 714}, 
        2: {}, 
        3: {'time': '2pm'}, 
        4: {'color': 'red'}, 
        5: {'color': 'green'}, 
        0: {}, 
        6: {}, 
        7: {}, 
        8: {}, 
        9: {}
        })
'''

# ----------------------------------------------------------------------------------------------------------------
# 边与属性
G.add_edge(1, 2, weight=4.7)
e = (2, 3)
G.add_edge(*e, weight=0.0)
G.add_edges_from([
        (1, 2, {"color": "blue"}),
        (1, 3, {"weight": 8.0})
        ])
G[1]  # AtlasView({2: {'weight': 4.7, 'color': 'blue'}, 3: {'weight': 8.0}})
G[1][2]  # {'weight': 4.7, 'color': 'blue'}
G.edges[1, 2]  # {'weight': 4.7, 'color': 'blue'}
G.edges[1, 2]["weight"] = 5.0  # 修改/增加边的属性

G.edges  # EdgeView([(1, 2), (1, 3), (2, 3)])
list(G.edges)  # [(1, 2), (1, 3), (2, 3)]
G.edges.data()
'''
EdgeDataView([
        (1, 2, {'weight': 5.0, 'color': 'blue'}), 
        (1, 3, {'weight': 8.0}), 
        (2, 3, {'weight': 0.0})
        ])
'''

# ----------------------------------------------------------------------------------------------------------------
# 一个图中的边可以合并到另一个图中
G.add_edges_from(H.edges)

# ----------------------------------------------------------------------------------------------------------------
# 输出结点和边的数量
G.number_of_nodes()
G.number_of_edges()

# ----------------------------------------------------------------------------------------------------------------
# 邻接表
G.adj
'''
AdjacencyView({
        1: {2: {'weight': 5.0, 'color': 'blue'}, 
            3: {'weight': 8.0}}, 
        
        2: {1: {'weight': 5.0, 'color': 'blue'}, 
            3: {'weight': 0.0}}, 
        
        3: {2: {'weight': 0.0}, 
            1: {'weight': 8.0}}, 
        
        4: {}, 
        5: {}, 
        0: {}, 
        6: {}, 
        7: {}, 
        8: {}, 
        9: {}
        })
'''
G.degree(1)  # 结点 1 的度
list(G.neighbors(1))  # [2, 3]

# ----------------------------------------------------------------------------------------------------------------
# 有向图
DG = nx.DiGraph()
DG.add_edge(2, 1)
DG.add_edge(1, 3)
DG.add_edge(2, 4)
DG.add_edge(1, 2)

DG.edges.data()
'''
OutEdgeDataView([
        (2, 1, {}), 
        (2, 4, {}), 
        (1, 3, {}), 
        (1, 2, {})
        ])
'''
list(DG.successors(1))  # 结点 1 的后序 [3, 2]
list(DG.predecessors(1))  # 结点 1 的前序 [2]

# ----------------------------------------------------------------------------------------------------------------
# 删除所有节点和边
G.clear()

# 从图形中删除元素
G.remove_node(4)
G.remove_nodes_from([5, 6])

G.remove_edge(1, 2)
G.remove_edges_from([
        (1, 2),
        (1, 3)
        ])

# ----------------------------------------------------------------------------------------------------------------
# 使用图形构造函数
G.add_edge(1, 2)
H = nx.DiGraph(G)
H.adj  # AdjacencyView({1: {2: {}}, 2: {1: {}}})
G.adj  # AdjacencyView({1: {2: {}}, 2: {1: {}}})

edgelist = [(0, 1), (1, 2), (2, 3)]
H = nx.Graph(edgelist)

adjacency_dict = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
H = nx.Graph(adjacency_dict)

# ----------------------------------------------------------------------------------------------------------------
# 访问边缘和邻居
FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
FG.adj  # 注意，对于无向图，邻接迭代可以看到每个边两次
'''
AdjacencyView({
        1: {2: {'weight': 0.125}, 
            3: {'weight': 0.75}}, 
        
        2: {1: {'weight': 0.125}, 
            4: {'weight': 1.2}}, 
        
        3: {1: {'weight': 0.75}, 
            4: {'weight': 0.375}}, 
        
        4: {2: {'weight': 1.2}, 
            3: {'weight': 0.375}}
        })
'''
FG.adj.items()

for (start_node, dct) in FG.adj.items():
    for (end_node, attr) in dct.items():
        weight = attr.get("weight")
        if weight < 0.5:
            print(f"({start_node}, {end_node}, {weight:.3})")

for (start_node, end_node, weight) in FG.edges.data("weight"):
    if weight < 0.5:
        print(f"({start_node}, {end_node}, {weight:.3})")

# ----------------------------------------------------------------------------------------------------------------
# 有向图
DG = nx.DiGraph()
DG.add_weighted_edges_from([
        (1, 2, 0.5),
        (3, 1, 0.75),
        (1, 4, 1.0)
        ])
DG.adj
'''
AdjacencyView({
        1: {2: {'weight': 0.5}}, 
        2: {4: {'weight': 1.0}},
        3: {1: {'weight': 0.75}}
        })
'''
DG.degree()  # 结点的度  DiDegreeView({1: 2, 2: 1, 3: 1})
DG.degree(1)  # 结点 1 的度  2
DG.degree(1, weight="weight")  # 结点 1 的权  1.25

DG.in_degree()  # 结点的入度  InDegreeView({1: 1, 2: 1, 3: 0})
DG.in_degree(1)  # 结点 1 的入度  1
DG.in_degree(1, weight="weight")  # 结点 1 的入权  0.75

DG.out_degree()  # 结点的出度  OutDegreeView({1: 1, 2: 0, 3: 1})
DG.out_degree(1)  # 结点 1 的出度  1
DG.out_degree(1, weight="weight")  # 结点 1 的出权  0.5

list(DG.successors(1))  # 结点 1 的后序结点  [2]
list(DG.predecessors(1))  # 结点 1 的前序结点  [3]
list(DG.neighbors(1))  # [2]

# ----------------------------------------------------------------------------------------------------------------
# 多重图
MG = nx.MultiGraph()
MG.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])
MG.adj
'''
MultiAdjacencyView({
        1: {2: {0: {'weight': 0.5}, 
                1: {'weight': 0.75}}}, 
        
        2: {1: {0: {'weight': 0.5}, 
                1: {'weight': 0.75}}, 
            3: {0: {'weight': 0.5}}}, 
        
        3: {2: {0: {'weight': 0.5}}}
        })
'''

# ----------------------------------------------------------------------------------------------------------------
# 使用常用图形格式读取存储在文件中的图形
nx.write_gml(MG, "path.to.file")
mygraph = nx.read_gml("path.to.file")

# ----------------------------------------------------------------------------------------------------------------
# 分析图形
G = nx.Graph()
G.add_edges_from([
        (1, 2), 
        (1, 3)
        ])
G.add_node("spam")
G.adj
'''
AdjacencyView({
        1: {2: {}, 3: {}}, 
        2: {1: {}}, 
        3: {1: {}}, 
        'spam': {}
        })
'''
list(nx.connected_components(G))  # [{1, 2, 3}, {'spam'}]
sorted(degree for (node, degree) in G.degree())  # [0, 1, 1, 2]
nx.clustering(G)  # {1: 0, 2: 0, 3: 0, 'spam': 0}

sp = dict(nx.all_pairs_shortest_path(G))
sp
'''
{
 1: {1: [1], 2: [1, 2], 3: [1, 3]},
 2: {2: [2], 1: [2, 1], 3: [2, 1, 3]},
 3: {3: [3], 1: [3, 1], 2: [3, 1, 2]},
 'spam': {'spam': ['spam']}
 }
'''

# ----------------------------------------------------------------------------------------------------------------
# 图形绘制
G = nx.petersen_graph()
G.adj
subax1 = plt.subplot(121)
nx.draw(G, with_labels=True, font_weight="bold")
subax2 = plt.subplot(122)
nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight="bold")

pos = nx.nx_agraph.graphviz_layout(G)
nx.draw(G, pos=pos)
write_dot(G, "file.dot")

