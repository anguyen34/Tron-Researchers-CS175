---
layout:	default
title:	Status Report
---

## Project Summary:

We are using ColosseumRL to study the effect of changing certain parameters (e.g. learning rate, number of layers, depth, number of epochs, etc) on the success of each policy that we implemented. Our goal is still to see which parameter configuration brings the most success and to see which parameters have the greatest effect on the agent in general. We are using four different approximators for Q-Learning, three of which are machine learning algorithms. The four approximators for Q-Learning act as the policies for our agents in order to determine the most influential parameter configuration per policy, after which we will use collected data to determine which parameters get the best results overall. This research approach will involve training five agents to play Tron. The beginning of the training for the machine learning approximators will involve four agents playing against each other in Tron, with each agent using the same policy, but one agent will have altered parameters. As we change the parameters we will study the effect on the agent's decisions and outcomes against its other similar agents. We will eventually find the optimal configuration of parameters that achieves the best results for that algorithm after which we will repeat this training process for the other two machine learning algorithms. For the Monte Carlo search tree the only parameter we will study is changing the max depth of the search tree to find an optimal max depth. Once we have optimized each agent, we will study the effect of these models interacting with different agents rather than the same. Our stretch goal is still to optimize the agents for playing against other agents, rather than similar agents.

## Approach:

For out project we make use of four different approximators, three of which are machine learning algorithms. For the machine learning algorithms we use a neural network, a random forest, and an ensemble. The fourth approximator used is a Monte Carlo search tree. For the neural network approximator, deep Q-learning is used to train the agent to minimize its loss according to the function:
![Loss is based on state, action, reward, and next state tuples,](https://github.com/anguyen34/Tron-Researchers-CS175/main/images/loss.png?raw=true)

## Evaluation:

## Remaining Goals and Challenges:

## Resources Used:
