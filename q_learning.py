import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

from reinforcelab.agents.agent import Agent

class QLearningAgent(Agent):
    def __init__(self, env, gamma, alpha):
        self.env = env
        
        self.state_size = len(self.env.custom_observation_space_dict.values())
        self.action_size = env.action_space.n
        dims = [self.state_size] + [self.action_size]
        self.qtable = np.zeros(dims)
        self.gamma = gamma
        self.alpha = alpha

    def __state2idx(self, state):
        idx = tuple([int(val) for val in state])
        return idx

    def act(self, state, epsilon=0.0):
        qvalues = self.qtable[state]
        action = np.argmax(qvalues)

        # Randomly choose an action with p=epsilon
        if np.random.random() < epsilon:
            action = np.random.choice(self.action_size)
        return action

    def update(self, state, action, reward, next_state, done):
        qvalue = self.qtable[state][action]
        next_qvalue = 0
        if not done:
            next_qvalue = np.max(self.qtable[next_state])

        # Compute td error
        td_error = reward + self.gamma * next_qvalue - qvalue

        # Update Q table
        new_val = qvalue + self.alpha * td_error
        self.qtable[state][action] = new_val

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "checkpoint.npy")
        np.save(filepath, self.qtable)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.npy")
        self.qtable = np.load(filepath)

