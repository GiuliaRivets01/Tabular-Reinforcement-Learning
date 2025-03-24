#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time

from Q_learning import q_learning
from SARSA import sarsa
from Nstep import n_step_Q
from MonteCarlo import monte_carlo
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma,
                             policy='egreedy',
                             epsilon=None, temp=None, smoothing_window=None, plot=False, n=5, eval_interval=500):
    returns_over_repetitions = []
    now = time.time()

    for rep in range(n_repetitions):  # Loop over repetitions
        if backup == 'q':
            print("Q-learning: ", rep)
            returns, timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot,
                                      eval_interval)
        elif backup == 'sarsa':
            print("SARSA: ", rep)
            returns, timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
        elif backup == 'nstep':
            print("n-step: ", rep)
            returns, timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                                          policy, epsilon, temp, plot, n, eval_interval)
        elif backup == 'mc':
            print("MC: ", rep)
            returns, timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                                             policy, epsilon, temp, plot, eval_interval)

        returns_over_repetitions.append(returns)

    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    print(len(returns_over_repetitions))
    for elem in returns_over_repetitions:
        print(len(elem))
    #input('ciao')
    #input('Ciao')
    learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)  # average over repetitions
    if smoothing_window is not None:
        learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    return learning_curve, timesteps


def experiment():
    ####### Settings
    # Experiment
    n_repetitions = 20
    smoothing_window = 9  # Must be an odd number. Use 'None' to switch smoothing off!
    plot = False  # Plotting is very slow, switch it off when we run repetitions

    # MDP
    n_timesteps = 50001  # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 1000
    max_episode_length = 100
    gamma = 1.0

    # Parameters we will vary in the experiments, set them to some initial values:
    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.05
    temp = 1.0
    # Back-up & update
    backup = 'q'  # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.1
    n = 5  # only used when backup = 'nstep'

    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning',
                     'sarsa': 'SARSA',
                     'mc': 'Monte Carlo',
                     'nstep': 'n-step Q-learning'}

    ####### Experiments

    #### Assignment 1: Dynamic Programming
    # Execute this assignment in DynamicProgramming.py
    optimal_episode_return = 82.7153448152381  # set the optimal return per episode you found in the DP assignment here

    # Assignment 2: Effect of exploration
    policy = 'egreedy'
    print("Gamma: ", gamma)
    gamma = 0.99
    epsilons = [0.03, 0.1, 0.3]
    learning_rate = 0.1
    backup = 'q'
    Plot = LearningCurvePlot(title='Exploration: $\epsilon$-greedy versus softmax exploration')
    Plot.set_ylim(-100, 100)
    for epsilon in epsilons:
        print("Policy: ", policy, ", epsilon: ", epsilon)
        learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length,
                                                             learning_rate,
                                                             gamma, policy, epsilon, temp, smoothing_window, plot, n,
                                                             eval_interval)
        Plot.add_curve(timesteps, learning_curve, label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
    policy = 'softmax'
    temps = [0.01, 0.1, 1.0]
    for temp in temps:
        print("Policy: ", policy, ",    temp: ", temp)
        learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length,
                                                             learning_rate,
                                                             gamma, policy, epsilon, temp, smoothing_window, plot, n,
                                                             eval_interval)
        Plot.add_curve(timesteps, learning_curve, label=r'softmax, $ \tau $ = {}'.format(temp))
    Plot.add_hline(optimal_episode_return, label="DP optimum")
    Plot.save('exploration.png')

    ## Assignment 3: Q-learning versus SARSA
    policy = 'egreedy'
    epsilon = 0.1  # set epsilon back to original value
    learning_rates = [0.03, 0.1, 0.3]
    backups = ['q', 'sarsa']
    Plot = LearningCurvePlot(title='Back-up: on-policy versus off-policy')
    Plot.set_ylim(-100, 100)
    for backup in backups:
        for learning_rate in learning_rates:
            learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length,
                                                                 learning_rate,
                                                                 gamma, policy, epsilon, temp, smoothing_window, plot,
                                                                 n, eval_interval)
            Plot.add_curve(timesteps, learning_curve,
                           label=r'{}, $\alpha$ = {} '.format(backup_labels[backup], learning_rate))
    Plot.add_hline(optimal_episode_return, label="DP optimum")
    Plot.save('on_off_policy.png')

    #### Assignment 4: Back-up depth
    policy = 'egreedy'
    epsilon = 0.05 # set epsilon back to original value
    learning_rate = 0.1
    backup = 'nstep'
    ns = [1,3,10]
    Plot = LearningCurvePlot(title = 'Back-up: depth')
    Plot.set_ylim(-100, 100)
    for n in ns:
        learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
        Plot.add_curve(timesteps,learning_curve,label=r'{}-step Q-learning'.format(n))
    backup = 'mc'
    learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                          gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
    Plot.add_curve(timesteps,learning_curve,label='Monte Carlo')
    Plot.add_hline(optimal_episode_return, label="DP optimum")
    Plot.save('depth.png')


    # Added part
    #### Tabular Q-learning
    # backup_q_learning = 'q'
    # learning_curve_q_learning, timesteps_q_learning = average_over_repetitions(backup_q_learning, n_repetitions,
    #                                                                              n_timesteps, max_episode_length,
    #                                                                              learning_rate,
    #                                                                              gamma, policy, epsilon, temp,
    #                                                                              smoothing_window, plot, n=5,
    #                                                                              eval_interval=eval_interval)
    # print(learning_curve_q_learning)
    #
    # #### n-step Q-learning with n = 1
    # backup_nstep = 'nstep'
    # n = 1
    # learning_curve_nstep, timesteps_nstep = average_over_repetitions(backup_nstep, n_repetitions, n_timesteps,
    #                                                                  max_episode_length, learning_rate,
    #                                                                  gamma, policy, epsilon, temp, smoothing_window,
    #                                                                  plot, n=n, eval_interval=eval_interval)
    #
    # #### Plotting
    # Plot = LearningCurvePlot(title='Tabular Q-learning vs n-step Q-learning (n = 1)')
    # Plot.set_ylim(-100, 100)
    # Plot.add_curve(timesteps_q_learning, learning_curve_q_learning, label='Tabular Q-learning')
    # Plot.add_curve(timesteps_nstep, learning_curve_nstep, label='n-step Q-learning (n = 1)')
    # Plot.save('q_learning_vs_nstep.png')




if __name__ == '__main__':
    experiment()

