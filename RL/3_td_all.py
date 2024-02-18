# -*- coding: utf-8 -*-
"""
TD控制
"""
import os
# import copy
# from functools import partial
# from collections import defaultdict
import numpy as np
import gym
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/RL"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 环境使用
env = gym.make('Taxi-v3')
print('观察空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('状态数量 = {}'.format(env.observation_space.n))
print('动作数量 = {}'.format(env.action_space.n))

state, _ = env.reset()
taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
print(taxirow, taxicol, passloc, destidx)
print('的士位置 = {}'.format((taxirow, taxicol)))
print('乘客位置 = {}'.format(env.unwrapped.locs[passloc]))
print('目标位置 = {}'.format(env.unwrapped.locs[destidx]))
# env.render()  # 与Jupyter Notebook不兼容
env.step(0)

# ----------------------------------------------------------------------------------------------------------------
# SARSA - 保守
class SARSA(object):
    def __init__(self, env, gamma=0.9, learning_rate=0.2, eps=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps = eps
        self.n_states = env.observation_space.n  # 状态集合
        self.n_actions = env.action_space.n  # 动作集合
        self.arr_q = np.zeros([self.n_states, self.n_actions])
    
    def decide(self, s):
        if np.random.uniform() > self.eps:  # 99%
            a = self.arr_q[s].argmax()
        else:  # 1%
            a = np.random.randint(self.n_actions)
        return a
    
    def learn(self, s, a, r, next_s, terminated, truncated, next_a):
        u = r + self.gamma * self.arr_q[next_s, next_a] * (1.0 - terminated)
        td_error = u - self.arr_q[s, a]
        self.arr_q[s, a] += self.learning_rate * td_error
        return 


def play_sarsa(env, agent, train=False, render=False):
    '''
    agent = SARSA(env, gamma=0.9, learning_rate=0.2, eps=0.01)
    '''
    total_r = 0
    s, _ = env.reset()  # Initialize S
    a = agent.decide(s)  # Choose A from S using policy derived from Q
    
    while True:
        if render:
            env.render()
            
        next_s, r, terminated, truncated, info = env.step(a)  # Take action A, observe R, S' (与环境交互，系统行为)
        total_r += r
        next_a = agent.decide(next_s)  # Choose A' from S' using policy derived from Q（软性策略b）
        
        if train:
            agent.learn(s, a, r, next_s, terminated, truncated, next_a)  # 学习Q-table
        if terminated or truncated:
            break
        
        s, a = next_s, next_a
    return total_r  

# 训练
agent = SARSA(env)
episodes = 3000
lst_total_r = []

for episode in range(episodes):
    total_r = play_sarsa(env, agent, train=True)
    lst_total_r.append(total_r)

plt.plot(lst_total_r)

# 测试
agent.epsilon = 0.0  # 取消探索
lst_total_r = [play_sarsa(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(lst_total_r), len(lst_total_r), np.mean(lst_total_r)))

# 显示最优策略估计
arr_pi = np.argmax(agent.arr_q, axis=1)

# ----------------------------------------------------------------------------------------------------------------
# Q-Learning - 激进（贪心）
class QLearning(object):
    def __init__(self, env, gamma=0.9, learning_rate=0.1, eps=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps = eps
        self.n_states = env.observation_space.n  # 状态集合
        self.n_actions = env.action_space.n  # 动作集合
        self.arr_q = np.zeros([self.n_states, self.n_actions])

    def decide(self, s):
        if np.random.uniform() > self.eps:
            a = self.arr_q[s].argmax()
        else:
            a = np.random.randint(self.n_actions)
        return a

    def learn(self, s, a, r, next_s, terminated, truncated):
        u = r + self.gamma * self.arr_q[next_s].max() * (1.0 - terminated)
        td_error = u - self.arr_q[s, a]
        self.arr_q[s, a] += self.learning_rate * td_error
        return 

def play_qlearning(env, agent, train=False, render=False):
    total_r = 0
    s, _ = env.reset()  # Initialize S
    
    while True:
        if render:
            env.render()
            
        a = agent.decide(s)  # Choose A from S using policy derived from Q
        next_s, r, terminated, truncated, info = env.step(a)  # Take action A, observe R, S'
        total_r += r
        
        if train:
            agent.learn(s, a, r, next_s, terminated, truncated)  # 学习Q-table
        if terminated or truncated:
            break
        s = next_s
    
    return total_r


# 训练
agent = QLearning(env)
episodes = 3000
lst_total_r = []

for episode in range(episodes):
    total_r = play_qlearning(env, agent, train=True)
    lst_total_r.append(total_r)

plt.plot(lst_total_r)

# 测试
agent.epsilon = 0.0  # 取消探索
lst_total_r = [play_sarsa(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(lst_total_r), len(lst_total_r), np.mean(lst_total_r)))

# 显示最优策略估计
arr_pi = np.argmax(agent.arr_q, axis=1)

# ----------------------------------------------------------------------------------------------------------------
# 期望 SARSA - 折中
class ExpectedSARSA(object):
    def __init__(self, env, gamma=0.9, learning_rate=0.1, eps=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps = eps
        self.n_states = env.observation_space.n  # 状态集合
        self.n_actions = env.action_space.n  # 动作集合
        self.arr_q = np.zeros([self.n_states, self.n_actions])

    def decide(self, s):
        if np.random.uniform() > self.eps:
            a = self.arr_q[s].argmax()
        else:
            a = np.random.randint(self.n_actions)
        return a

    def learn(self, s, a, r, next_s, terminated, truncated):
        v = self.arr_q[next_s].mean() * self.eps + self.arr_q[next_s].max() * (1.0 - self.eps)
        u = r + self.gamma * v * (1.0 - terminated)
        td_error = u - self.arr_q[s, a]
        self.arr_q[s, a] += self.learning_rate * td_error
        return 

# 训练
agent = ExpectedSARSA(env)
episodes = 3000
lst_total_r = []

for episode in range(episodes):
    total_r = play_qlearning(env, agent, train=True)
    lst_total_r.append(total_r)

plt.plot(lst_total_r)

# 测试
agent.epsilon = 0.0  # 取消探索
lst_total_r = [play_sarsa(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(lst_total_r), len(lst_total_r), np.mean(lst_total_r)))

# 显示最优策略估计
arr_pi = np.argmax(agent.arr_q, axis=1)
    
# ----------------------------------------------------------------------------------------------------------------
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

