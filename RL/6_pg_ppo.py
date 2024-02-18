# -*- coding: utf-8 -*-
"""
https://github.com/Lukaschen1986/Youtube-Code-Repository/tree/master/ReinforcementLearning
https://www.youtube.com/watch?v=hlv79rcHws0&t=2642s
https://www.bilibili.com/video/BV1jg411M7HB/?spm_id_from=333.880.my_history.page.click&vd_source=fac9279bd4e33309b405d472b24286a8
"""
import os
import numpy as np
import gym
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/RL"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# memory
class PPOMemory(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

# ----------------------------------------------------------------------------------------------------------------
# actor
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir=path_model):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=(0.9, 0.999), eps=10**-8)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        th.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(th.load(self.checkpoint_file))

# ----------------------------------------------------------------------------------------------------------------
# critic
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir=path_model):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=(0.9, 0.999), eps=10**-8)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        th.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(th.load(self.checkpoint_file))

# ----------------------------------------------------------------------------------------------------------------
# agent
class Agent(object):
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, lamb=0.95,
                 policy_clip=0.2, batch_size=64, epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.epochs = epochs
        self.lamb = lamb
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = th.tensor([observation], dtype=th.float).to(self.actor.device)
        
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        
        probs = th.squeeze(dist.log_prob(action)).item()  # log_Pi(a|s) old
        action = th.squeeze(action).item()
        value = th.squeeze(value).item()
        return action, probs, value
    
    def learn(self):
        for _ in range(self.epochs):
            # 生成一幕数据
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()
                
            # 评估 advantage function
            V = vals_arr
            A = np.zeros(len(reward_arr), dtype=np.float32)  # init advantage function
            
            for t in range(len(reward_arr)-1):
                discount = 1
                dt = 0
                for k in range(t, len(reward_arr)-1):
                    dt += discount*(reward_arr[k] + self.gamma*V[k+1]*(1-int(dones_arr[k])) - V[k])  # 论文公式11-12
                    discount *= self.gamma * self.lamb
                A[t] = dt
                
            A = th.tensor(A).to(self.actor.device)
            V = th.tensor(V).to(self.actor.device)
            
            # 计算策略更新
            for batch in batches:
                states = th.tensor(state_arr[batch], dtype=th.float).to(self.actor.device)
                old_probs = th.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = th.tensor(action_arr[batch]).to(self.actor.device)
                
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = th.squeeze(critic_value)
                
                new_probs = dist.log_prob(actions)  # log_Pi(a|s) new
                rt = new_probs.exp() / old_probs.exp()  # 论文公式6
                weighted_probs = rt * A[batch]
                weighted_probs_clip = th.clamp(rt, 1-self.policy_clip, 1+self.policy_clip) * A[batch]
                
                actor_loss = -1.0 * th.min(weighted_probs, weighted_probs_clip).mean()  # 论文公式7
                critic_loss = ((A[batch] + V[batch] - critic_value)**2).mean()
                total_loss = actor_loss + 0.5*critic_loss  # 论文公式9
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()
        return 

# ----------------------------------------------------------------------------------------------------------------
# plot
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
    

if __name__ == "__main__":
    env = gym.make("CartPole-v0")  # pip install gym==0.25
    
    N = 20
    batch_size = 5
    epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                  alpha=alpha, epochs=epochs, 
                  input_dims=env.observation_space.shape)
    
    n_games = 100
    figure_file = os.path.join(path_model, "cartpole.png")
    best_score = env.reward_range[0]
    score_history = []
    
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
 
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print(f"episode {i} score {score:.1f} avg score {avg_score:.1f} time_steps {n_steps} learning_steps {learn_iters}")
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
