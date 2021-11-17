---
layout: default
title:  Home
---

## Source code: 

[https://github.com/anguyen34/Tron-Researchers-CS175](https://github.com/anguyen34/Tron-Researchers-CS175)

## ColosseumRL: 

[https://colosseumrl.igb.uci.edu/](https://colosseumrl.igb.uci.edu/)

## Project Summary:

We study the effects of changing various parameters across four different functional approximators for Q-Learning in the ColosseumRL Tron game. Our goal is to see which parameters are the most influential in determining the performance of our policies in Tron. A secondary goal is to determine a fine tuned set of parameters for each policy using data from our primary goal, which plays Tron optimally against other agents using the same policy. Once we have determined the optimal parameters for each policy we intend to have different policies play against each other to determine which policy is the most effective at playing Tron. The five approximators we use are a neural network, a random forest, an ensemble, a linear combination of features, and a Monte Carlo search tree.

## Reports:

- [Proposal](proposal.html) 
- [Updated Proposal](proposalUpdate.html)
- [Status](status.html)
- [Final](final.html)

## ColosseumRL's Tron:

The game of Tron as implemented in ColosseumRL is based on the light cycle subgame from the original Tron, which itself is derived from Snake games. In the Tron Light Cycle game the goal is to force opposing AI to crash into the wall or into one's own trail, while simultaneously avoiding the enemy's trails.

![](images/colrl_tron.png?raw=true)

## Original Tron Light Cycle Game:
<iframe width="560" height="315" src="https://www.youtube.com/watch?v=XEp8G2HtDJM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
