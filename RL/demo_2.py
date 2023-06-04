# -*- coding: utf-8 -*-
"""
https://www.bilibili.com/video/BV1R84y1T7s8/?spm_id_from=333.999.0.0&vd_source=fac9279bd4e33309b405d472b24286a8
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
# 策略迭代
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
arr_pi = np.random.randint(low=0, high=n_actions, size=n_states, dtype=int)  # 每个状态s所对应的策略，初始化随机性策略
arr_q = np.zeros([n_states, n_actions])

# ----------------------------------------------------------------------------------------------------------------
# 策略迭代：策略评估（pi -> v）
gamma = 0.9
n_iters = 1000
eps = 10**-8

def policy_evaluation(arr_pi):
    for i in range(n_iters):
        prev_arr_v = copy.deepcopy(arr_v)
        
        # 遍历状态索引（S = s）
        for s in range(n_states):
            a = arr_pi[s]  # 获取确定性策略
            v = 0
            for (prob, next_s, r, flag) in env.P[s][a]:
                v += prob * (r + gamma * prev_arr_v[next_s])  # 1.2-对应v(s)的积分公式
            arr_v[s] = v  # 1.2-贝尔曼期望方程
        
        if np.max(np.abs(prev_arr_v - arr_v)) < eps:
            break
    
    return arr_v

# ----------------------------------------------------------------------------------------------------------------
# 策略迭代：策略改进（q -> pi'）
def policy_update(arr_v):
    # 遍历状态索引（S = s）
    for s in range(n_states):
        # 遍历动作索引（A = a）
        for a in range(n_actions):
            for (prob, next_s, r, flag) in env.P[s][a]:
                arr_q[s, a] += prob * (r + gamma * arr_v[next_s])  # 1.3-对应q(s, a)的积分公式
    
    arr_pi = np.argmax(arr_q, axis=1)  # 2.4-贪心策略
    return arr_pi

# ----------------------------------------------------------------------------------------------------------------
# 策略迭代
while True:
    arr_v = policy_evaluation(arr_pi)
    arr_pi_new = policy_update(arr_v)
    
    if np.array_equal(arr_pi_new, arr_pi):
        break
    else:
        arr_pi = arr_pi_new

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
        
        


