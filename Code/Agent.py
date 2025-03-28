#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax

class BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        if policy == 'greedy':
            # TO DO: Add own code
            #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
            a = argmax(self.Q_sa[s])


        elif policy == 'egreedy':
            # TO DO: Add own code
            # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            else:
                if np.random.rand() < epsilon: # Select a random action with probability e
                    a = np.random.randint(0, self.n_actions)
                else: # Take greedy action
                    a = argmax(self.Q_sa[s])

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            else:
                ''' calculate the probabilities of each action using softmax function and
                choose an action based on these probabilities.'''
                probability = softmax(self.Q_sa[s], temp)
                a = np.random.choice(np.arange(self.n_actions), p=probability)

        return a

    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

