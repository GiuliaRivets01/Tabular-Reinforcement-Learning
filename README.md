# Tabular-Reinforcement-Learning
Assignment 1 of the course *Reinforcement Learning*

In this assignment we studied a range of basic principles in tabular, value-based reinyforcement learning:
1. **Dynamic Programming**, where full access to a model of the environment is assumed.
2. **Model-free RL**, where we do not have access to a model, but we can only permanently execute actions from a state, and have to continue from the resulting next state.

## Environment
The environment used for this assignment is the *Stochastic Windy Gridworld*, which consists of a 10x7 grid, where at each cell we can move up, down, left or right.

## File Organization
*Agent.py* contains the Agent baseclass, in which we implemented the exploration methods and where an evaluate method to run greedy evaluation episodes throughout training is present. 

*Dynamic_Programming.py*  contains placeholder classes and functions for the Dynamic Programming experiments.

*Environment.py*: generates the environment. 

*Experiment.py* contains the experiment of all the considered algorithms (Q-Learning, SARSA, Monte Carlo RL and n-step Q-Learning).

*Helper.py* contains some helper classes for plotting and smoothing. 

*MonteCarlo.py* contains placeholder classes and functions for the Monte Carlo RL implementation.

*Nstep.py* contains placeholder classes and functions for the n-step Q-learning implementation.

*Q-learning.py* contains placeholder classes and functions for the Q-learning implementation.

*SARSA.py* contains placeholder classes and functions for the SARSA implementation.