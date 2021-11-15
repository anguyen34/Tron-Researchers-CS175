---
layout:	default
title:	Status Report
---

## Project Summary:

We are using ColosseumRL to study the effect of changing certain parameters (e.g. learning rate, number of layers, depth, number of epochs, etc) on the success of each policy that we implemented. Our goal is still to see which parameter configuration brings the most success and to see which parameters have the greatest effect on the agent in general. We are using four different approximators for Q-Learning, three of which are machine learning algorithms. The four approximators for Q-Learning act as the policies for our agents in order to determine the most influential parameter configuration per policy, after which we will use collected data to determine which parameters get the best results overall. This research approach will involve training five agents to play Tron. The beginning of the training for the machine learning approximators will involve four agents playing against each other in Tron, with each agent using the same policy, but one agent will have altered parameters. As we change the parameters we will study the effect on the agent's decisions and outcomes against its other similar agents. We will eventually find the optimal configuration of parameters that achieves the best results for that algorithm after which we will repeat this training process for the other two machine learning algorithms. For the Monte Carlo search tree the only parameter we will study is changing the max depth of the search tree to find an optimal max depth. Once we have optimized each agent, we will study the effect of these models interacting with different agents rather than the same. Our stretch goal is still to optimize the agents for playing against other agents, rather than similar agents.

## Approach:

For our project we make use of four different approximators, three of which are machine learning algorithms. For the machine learning algorithms we use a neural network, a random forest, and an ensemble. The fourth approximator used is a Monte Carlo search tree. All four approximators use a greedy epsilion policy, where epsilon determines the probability a random move is chosen instead of a predicted move. For the neural network approximator, deep Q-learning is used to train the agent to minimize its loss according to the function:
![](/images/loss.png?raw=true)

Then the move taken by the agent is in Tron is based on the computed action from the neural network. In the random forest we use a regressor to predict Tron reward values based on the current board state, then the move with the greatest predicted reward value is chosen. The random forest regressor agent is trained on the various game states and corresponding rewards of its past games. The ensemble is a voting regressor composed of an AdaBoost regressor, a K-Nearest neighbors regressor, a Multi-Layer perceptron regressor, and a support vector machine regressor. Training and choosing moves for the ensemble agent is the same as the random forest regressor. The fourth and final approximator used is a Monte Carlo search tree, which chooses rewards based on the calculated rewards at each leaf node in the search tree.

To test our four approximators we have our approximators play the game of Tron as it was implemented in ColosseumRL. Each player in Tron possesses three actions they can take at any time, going forward, left, or right. The reward function for a given board state and the actions taken by a player was already implemented in ColosseumRL. The reward function gives players a negative reward for dying, an increased reward for winning, and a default positive reward for not dying. The possible states for a given action is essentially all the possible combinations of moves made by the other players.

To determine the most influential and optimal parameters for an approximator, we will use code that will loop through various combinations of parameters for one agent, while the three other agents use the default parameters for the policy. For each parameter combination hundreds of games will be run where the agents is learning over all the games. As we loop through the combinations of parameters we will record data on the parameters used, the cumulative rewards for the modified agent, and the cumulative rewards of the unmodified agents. From the data we will determine which parameters caused the most learning and which parameters gave the best gameplay performance relative to the default parameters. Then when comparing different policies against each other we will setup four agents, each using a different approximator with its optimal parameters. We will then run through multiple games again, where each agent is learning. At the end we will compare the performances of the four agents/approximators over the course of the games and as they learned.

## Evaluation:

## Remaining Goals and Challenges:

## Resources Used:

- [ColosseumRL](https://colosseumrl.igb.uci.edu/)
- ColosseumRL setup from course website
- [SciKit-Learn](https://scikit-learn.org/stable/)
- [Ray](https://www.ray.io/)
- [Gym](https://gym.openai.com/)
