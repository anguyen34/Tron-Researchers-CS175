---
layout:	default
title:	Status Report
---

## Project Summary:

We are using ColosseumRL to study the effect of changing certain parameters (e.g. learning rate, number of layers, depth, number of epochs, etc) on the success of each policy that we implemented. Our goal is still to see which parameter configuration brings the most success and to see which parameters have the greatest effect on the agent in general. We are using four different approximators for Q-Learning, three of which are machine learning algorithms. The four approximators for Q-Learning act as the policies for our agents in order to determine the most influential parameter configuration per policy, after which we will use collected data to determine which parameters get the best results overall. This research approach will involve training five agents to play Tron. The beginning of the training for the machine learning approximators will involve four agents playing against each other in Tron, with each agent using the same policy, but one agent will have altered parameters. As we change the parameters we will study the effect on the agent's decisions and outcomes against its other similar agents. We will eventually find the optimal configuration of parameters that achieves the best results for that algorithm after which we will repeat this training process for the other two machine learning algorithms. For the Monte Carlo search tree the only parameter we will study is changing the max depth of the search tree to find an optimal max depth. Once we have optimized each agent, we will study the effect of these models interacting with different agents rather than the same. Our stretch goal is still to optimize the agents for playing against other agents, rather than similar agents.

## Approach:

For our project we make use of four different approximators, three of which are machine learning algorithms. For the machine learning algorithms we use a neural network, a random forest, and an ensemble. The fourth approximator used is a Monte Carlo search tree. All four approximators use a greedy epsilion policy, where epsilon determines the probability a random move is chosen instead of a predicted move. For the neural network approximator, deep Q-learning is used to train the agent to minimize its loss according to the function:
![](images/loss.png?raw=true)

Then the move taken by the agent is in Tron is based on the computed action from the neural network. In the random forest we use a regressor to predict Tron reward values based on the current board state, then the move with the greatest predicted reward value is chosen. The random forest regressor agent is trained on the various game states and corresponding rewards of its past games. The ensemble is a voting regressor composed of an AdaBoost regressor, a K-Nearest neighbors regressor, a Multi-Layer perceptron regressor, and a support vector machine regressor. Training and choosing moves for the ensemble agent is the same as the random forest regressor. The fourth and final approximator used is a Monte Carlo search tree, which chooses rewards based on the calculated rewards at each leaf node in the search tree.

To test our four approximators we have our approximators play the game of Tron as it was implemented in ColosseumRL. Each player in Tron possesses three actions they can take at any time, going forward, left, or right. The reward function for a given board state and the actions taken by a player was already implemented in ColosseumRL. The reward function gives players a negative reward for dying, an increased reward for winning, and a default positive reward for not dying. The possible states for a given action is essentially all the possible combinations of moves made by the other players.

To determine the most influential and optimal parameters for an approximator, we will modify the parameters of one agent, while the three other agents use the default parameters for the policy. We will loop through various values for the parameters and test several parameters to see which ones are influential. For each parameter value hundreds of games will be run where the agents is learning over all the games. As we loop through the values for the parameters we will record data on the parameters used, the cumulative rewards for the modified agent, and the cumulative rewards of the unmodified agents. From the data we will determine which parameters caused the most learning and which parameters gave the best gameplay performance relative to the default parameters. Then when comparing different policies against each other we will setup four agents, each using a different approximator with its optimal parameters. We will then run through multiple games again, where each agent is learning. At the end we will compare the performances of the four agents/approximators over the course of the games and as they learned.

## Evaluation:

Quantitative:  
  To evaluate the quantitative change in our approximators we will use graph plots and heatmaps. We intend to record data on the cumulative reward per game for the agent with modified parameters and the difference in cumulative reward between the modified agent and the averaged rewards of the three other agents. From this data we will generate plots or heatmaps for each parameter that is being changed. These plots/heatmaps will showcase how rewards and differences in rewards between agents change as parameters are varied. Using these plots we can determine which parameters had the greatest effect as the parameter value was changed and the optimal value for a parameter if it exists. Using the optimal values across multiple parameters we can pick the parameter values to use for an optimal Tron agent.
The reward plots for the baseline parameter set of the ensemble and random forest model are shown below:  
![](images/agent_forest_data_baseline.png?raw=true)
![](images/agent_forest_data_delta_reward.png?raw=true)
![](images/agent_ensemble_data_baseline.png?raw=true)
![](images/agent_ensemble_data_delta_reward.png?raw=true)  
For the baseline plots of observed agent reward vs other agent's average reward, the data is very inconsistent and tends to oscillate around 0 which is the predicted value for this since all agents have the same parameters for these plots and these parameters have not been changed. The plots showing the baseline cumulative reward for the observed agent are a useful visualization of the average performance of the model without changing parameters and leaving only default parameters.

Qualitative:  
  To evaluate the qualitative performance in our approximators we will observe the gameplay of our agent for which we change parameters in order to see and take notes of any observed performance increases and more intelligent behavior. This type of evaluation can be seen in our video summary which showcases some of the agent's gameplay. This gameplay can be observed for perceived increases in score as a secondaory metric for evaulating our models, in addition to the numbers given from our quantitative analysis.

## Remaining Goals and Challenges:

In the current state of our code all four approximators have been implemented and work for playing Tron. For each approximator, an agent using the approximator and defined parameters plays against three other agents using the same approximator with default parameters. We still need to finish implementing the code to run through all of the parameter combinations and to collect data. But, we believe this is a small amount of work based on prior experience with past coursework. Furthermore, we still need to implement code to run a game of Tron where each of the four players each use a different approximator. But, this should also be straightforward to implement as we have the code needed for all four approximators already.

The current goal is to finish implementing the code needed to collect and plot data on parameters. Following that each group member will by assigned an approximator to collect data for and afterwards we will discuss the results of the data collection, including determining the most influential parameters and optimal parameters. We believe this will be completed by the start of the week of Thanksgiving. Afterwards, we will implement code to have the agents with differing approximators play against each other and we will also collect data on which approximators result in better performing agents. We expect to be finished with collecting all of our results and to have reached our conclusions by the weekend before project presentations begin.

Based on our experiences of implementing and testing our previous algorithms, we expect one possible challenge is runtimes. Our neural network approximator takes significantly longer to learn when compared to the other two machine learning algorithms. Furthermore, the neural network algorithm is more resource intensive to run compared to the other approximators. To accomodate for the greater difficulty in running the neural network we plan on giving more time to collecting data on the neural network and have multiple people to collect data. This may affect our ability to run a large amount of games in a concise timeframe with the neural network. Another potential challenge is determining which parameters can produce the most change in learning and determing whether the range of tested parameter values is sufficient to see a change in gameplay performance. This could be the most crippling challenge as insufficient testing can result in us reaching the wrong conclusion. To combat this challenge, we simply plan on using our prior experiences with our approximators to determine potentially influential parameters and a good range of parameter values. Challenges could also occur when having agents with different approximators play against each other. One such challenge is a long runtime to play the hundreds of games, as some approximators take longer to learn than others.

## Resources Used:

- [ColosseumRL](https://colosseumrl.igb.uci.edu/)
- ColosseumRL setup/example from course website
- [SciKit-Learn](https://scikit-learn.org/stable/)
- [Ray](https://www.ray.io/)
- [Gym](https://gym.openai.com/)

## Video Summary:

<iframe width="560" height="315" src="https://www.youtube.com/embed/NHJ0bFcvIUI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
