# Agent that learns how to play by using a simple Policy Gradient method

import gym
import gym_abadia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# Environment definitions
abadia_inputdim = (1, 24, 24)
abadia_actions = 7
eps = np.finfo(np.float32).eps.item()

# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepro_frame(x):
    """Preprocess a game frame to make it amenable for learning"""
    return np.reshape(x["rejilla"], abadia_inputdim)


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted rewards"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) + eps
    return discounted_r


class Policy(nn.Module):
    """Pytorch CNN implementing a Policy"""
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.head = nn.Linear(288, abadia_actions)

        self.saved_log_probs = []

    def forward(self, x):
        x = F.relu(self.bn1((self.conv1(x))))
        x = F.relu(self.bn2((self.conv2(x))))
        return F.softmax(self.head(x.view(x.size(0), -1)), dim=1)

    def select_action(self, state):
        state = state.float().unsqueeze(0)
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()


def runepisode(env, policy, steps=2000, render=False):
    observation = env.reset()
    x = prepro_frame(observation)
    observations = []
    rewards = []
    rawframes = []

    for _ in range(steps):
        if render:
            env.render()
        x = torch.tensor(x).to(device)
        action = policy.select_action(x)
        observation, reward, done, info = env.step(action)
        x = prepro_frame(observation)
        observations.append(x)
        rewards.append(reward)
        rawframes.append(observation)
        if done:
            break

    return rewards, observations, rawframes


def train(render=False, checkpoint='policygradient.pt'):
    env = gym.make('Abadia-v0')
    try:
        policy = torch.load(checkpoint)
        print("Resumed checkpoint {}".format(checkpoint))
    except:
        policy = Policy()
        print("Created policy network from scratch")
    print(policy)
    policy.to(device)
    print("device: {}".format(device))
    optimizer = optim.RMSprop(policy.parameters(), lr=1e-4)

    episode = 0
    while True:
        # Gather samples
        rewards, observations, rawframes = runepisode(env, policy, render=render)
        print("Total reward for episode {}: {}".format(episode, np.sum(rewards)))
        drewards = discount_rewards(rewards)
        # Update policy network
        policy_loss = [-log_prob * reward for log_prob, reward in zip(policy.saved_log_probs, drewards)]
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.saved_log_probs[:]

        episode += 1
        # Save policy network from time to time
        if not episode % 100:
            torch.save(policy, checkpoint)

if __name__ == "__main__":
    train(render=False)
