# -*- coding: utf-8 -*-
"""
价值迭代
https://www.bilibili.com/video/BV1Yo4y1n7om/?spm_id_from=333.880.my_history.page.click&vd_source=fac9279bd4e33309b405d472b24286a8
"""
import os
import copy
import numpy as np
import gym


# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/RL"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 价值迭代
"""
A
0: 左
1: 下
2: 右
3: 上
"""
env = gym.make("FrozenLake-v1", render_mode="human")
env.reset()
env.render()

action = env.action_space.sample()
print(action)
env.step(action)
env.render()

# 价值函数（每个状态s所对应的价值量v）
n_states = env.observation_space.n  # 状态集合
n_actions = env.action_space.n  # 动作集合
arr_v = np.zeros(n_states)  # 每个状态s所对应的价值量
arr_pi = np.zeros(n_states, dtype=int)  # 每个状态s所对应的策略
print(arr_v)
print(arr_pi)

# 价值迭代
gamma = 0.9
n_iters = 1000
eps = 10**-8

for i in range(n_iters):
    prev_arr_v = copy.deepcopy(arr_v)
    
    # 遍历状态索引（S = s）
    for s in range(n_states):
        lst_q = []  # 给定s，会产生n_actions个q，依次存入lst_q
        
        # 遍历动作索引（A = a）
        for a in range(n_actions):
            q = 0  # 给定s和a，会产生一组q
            for (prob, next_s, r, flag) in env.P[s][a]:  # env.P为状态转移矩阵；prob为动态特性
                q += prob * (r + gamma * prev_arr_v[next_s])  # 1.3-对应q(s, a)的积分公式
            lst_q.append(q)
        
        arr_v[s] = np.max(lst_q)  # 1.3-贝尔曼最优方程
        arr_pi[s] = np.argmax(lst_q)  # 2.4-贪心策略
    
    if np.max(np.abs(prev_arr_v - arr_v)) < eps:
        break

print(arr_v)
print(arr_pi.reshape(4, 4))

# 测试
env.reset()
env.render()
flag = False

while not flag:
    s = env.s
    a = arr_pi[s]
    next_s, r, flag, truncated, info = env.step(a)
    env.render()
    print(s)
    
    

