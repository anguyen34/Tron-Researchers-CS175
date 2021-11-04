# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import numpy as np
from colosseumrl.envs.blokus import BlokusEnvironment as be

import gym
from gym.spaces import Dict, Discrete, Box

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG



class NeuralAgent:
  '''
  Agent that uses a neural network.
  '''
  def __init__(self):
    pass
