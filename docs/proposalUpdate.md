---
layout:	default
title:	Proposal Update
---

## Project Summary
We will be using ColosseumRL to study the effect of changing certain parameters (e.g. learning rate) on the success of each policy that we will use. The goal is not only to see which parameter configuration brings the most success but also to see which parameters have the greatest effect on the agent in general. We will use four different machine learning algorithms as the policy for our agents in order to determine the best/most influential parameter configuration per policy, after which we will test which optimized policy gets the best results overall. This research approach will involve training 4 agents to play Blockus. The beginning of the training will involve each of the 4 agents using the same algorithm as we change the parameters and study their effect on the agent's decisions and outcomes. We will eventually find the optimal configuration that achieves the best results for that algorithm after which we will repeat this process for the other 3 algorithms. Once we have optimized each model, we will study the effect of these models interacting with different models rather than the same. A stretch goal for this may be once again changing parameters to see which ones effect the agent's decisions and outcomes the most when playing against different policies.


## AI/ML Algorithms
We will be using reinforcement learning with the following 4 approximators:
- Random Forests
- Neural Network
- Monte Carlo Search Tree (most simple therefore our baseline)
- Ensemble (containing all of the above)


## Evaluation Plan
**Quantitative:**
As we are studying the effect of modifying certain parameters on the game, our metric will be the change in each parameter between games and how much that parameter change affected the score between games for each policy. For our secondary goal of finding the most optimal parameter configuration for each model, our metric will be the change in the score and number of turns taken to expend all pieces for an agent. The baseline for our primary goal will be the assumption all parameters will affect the agent decision/game outcome equally, and we will be researching if this is true.

**Qualitative:**

 

## Appointment with the Instructor
