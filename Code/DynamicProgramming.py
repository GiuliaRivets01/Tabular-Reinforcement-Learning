#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.threshold = threshold
        self.start_location = (0, 3)
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        a = argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        newQ_sa = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1)))
        self.Q_sa[s, a] = newQ_sa
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

     # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    ''' First initialize an agent, and then sweep through the state space, 
    # each time calling the model and then updating the agent, until convergence. '''
    i = 0
    while True:
        lambda_var = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                p_sas, r_sas = env.model(s, a)
                x = QIagent.Q_sa[s, a] # Store current estimate
                QIagent.update(s, a, p_sas, r_sas)

                lambda_var = max(lambda_var, abs(x - QIagent.Q_sa[s, a])) # Update max error

        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=False,step_pause=0.5)
        max_error = lambda_var
        if i == 0 or i == 8 or i==16:
            input("Enter to continue")
        # Plot current Q-value estimates & print max error
        i += 1
        print("Q-value iteration, iteration {}, max error {}".format(i, max_error))

        if lambda_var < threshold:
            break


    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)

    ''' Parte 1c punto 3 
        Fomrula: https://towardsdatascience.com/convergence-of-reinforcement-learning-algorithms-3d917f66b3b7 '''
    Q_opt_sa = QIagent.Q_sa[env.start_location]
    V_opt_s = np.max(Q_opt_sa)
    print("\nThe converged optimal value at the start is: ", V_opt_s)

    
    # view optimal policy
    done = False
    s = env.reset()

    i = 0
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)

        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=False,step_pause=0.1)
        s = s_next
        i += 1

    # TO DO: Compute mean reward per timestep under the optimal policy
    ''' Parte 1c punto 4 
        a) Avg # steps agent needs to take to reach goal based on:
        - optimal value at start state
        - magnitude of final reward
        - magnitude of reward on every other step '''
    # Calculate mean reward per timestep under the optimal policy
    rewards_per_timestep = []
    for _ in range(10000):
        done = False
        s = env.reset()
        total_reward = 0
        while not done:
            a = QIagent.select_action(s)
            s_next, r, done = env.step(a)
            total_reward += r
            s = s_next
        # Append the quantity return / (100 - return) to the list
        rewards_per_timestep.append(total_reward / (100 - total_reward))

    # Compute the mean return per timestep
    mean_return_per_timestep = np.mean(rewards_per_timestep)
    print("Mean return per timestep under the optimal policy:", mean_return_per_timestep)


if __name__ == '__main__':
    experiment()
