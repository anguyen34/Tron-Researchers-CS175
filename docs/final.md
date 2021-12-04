---
layout:	default
title:	Final Report
---

## Video:

## Project Summary:
For our project, we conducted research of various parameters across three different machine learning algorithms and Monte Carlo search, all of which were used as functional approximators for reinforcement Q-learning. We studied how changing the values of the selected parameters would impact the success of each policy in simulated games against other policies with the same algorithm, but using parameters we defined as the baseline parameters. The platform used for conducting our research was the Tron environment from ColosseumRL, a multiagent reinforcement learning environment comprising several different games. The game we chose, Tron as implemented in ColosseumRL is the Tron light cycles subgame, which is a variation of the snake game concept, but here the goal is to force opponents into walls or trails, while simultaneously avoiding them. However, unlike the original Tron light cycles subgame which was a player versus three AI, the Tron environment we used was four AI playing against each other. 

Our primary goal was to study the effect of changing various parameters of our reinforcement Q-learning functional approximators on our learning rates and overall success in the game for each policy. The three machine learning algorithms used as functional approximators for the Q-learning Greedy Epsilon policy are Random Forest, Neural Network, Monte Carlo Search, and Ensemble which contains a MultiLayer Perceptron Network, Adaboost Regressor, Support Vector Machine, and K-Nearest Neighbor Regressor. To satisfy our primary goal we researched the performance of each functional approximator while changing parameters such as number of hidden layers and nodes per layer for the Neural Network or maximum number of leaves and maximum depth for the Random Forest. Evaluating the performance of a model for each parameter configuration gives insight into the most impactful parameters on the model’s gameplay which is exactly what we are researching.

Our secondary goal was to identify which parameters of high impact were the most beneficial to our AI and to have our four functional approximators with their optimal parameters face off each other. Beyond just observing the impacts of changing certain parameters in a particular functional approximator, we also wanted to observe how the functional approximators compared with each other. We expected our neural network approximator to perform the best, since while researching for functional approximators to use for our project, we learned that neural networks are the most commonly used approximators for reinforcement learning, compared to an approximator like Random Forest. By comparing our neural network approximator with our other ones, we wanted to see if this conclusion also applied to AI that were training on and playing Tron.

What motivated our idea behind the project was the idea that games are a traditional field that AI are used in, such as Tron originally using AI as opponents. Developing better AI could result in more challenging and interesting AI opponents for Tron and other games. Machine learning is a core part of AI, since it can help produce solutions to problems where the solutions are hard to describe. An instance of this is Tron, where the solution to the game is winning against four other players. A solution must produce a series of actions for the player in which they are the winner, but also account for the actions of other players, and the numerous potential board states. Determining machine learning parameters that could result in a faster learning AI or a more successful AI are beneficial to actually producing better AI within a quicker time frame. Through researching the various machine learning parameters across several algorithms, we hoped to find some of the parameters that could lead to faster learning or more successful AI.


## Approaches:

## Evaluation:

## References:
