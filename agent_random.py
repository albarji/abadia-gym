# Copy of the agentv1 but with correct data processing
# It just performs random actions

import gym
import gym_abadia
import numpy as np

episodes = 20
episode_length = 2000

env = gym.make('Abadia-v0')
for i_episode in range(episodes):
    observation = env.reset()
    totalreward = 0
    for t in range(episode_length):
        env.render(mode="human")
        action = env.action_space.sample()

        print("Next Action: {}\n".format(action))
        observation, reward, done, info = env.step(action)
        totalreward += reward

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    print("Total reward at end of episode: {}".format(totalreward))
