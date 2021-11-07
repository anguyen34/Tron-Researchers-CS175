# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import numpy as np
import random

from matplotlib import cm
from time import sleep
from colosseumrl.envs.tron import TronGridEnvironment, TronRender
from colosseumrl.envs.tron.rllib import TronRaySinglePlayerEnvironment

import gym
from gym.spaces import Dict, Discrete, Box

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.rllib.env.multi_agent_env import MultiAgentEnv


from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models import ModelCatalog

SEED = 12345
np.random.seed(SEED)

# A full free-for-all version of tron
class TronRayEnvironment(MultiAgentEnv):
    action_space = Discrete(3)

    def __init__(self, board_size=15, num_players=4):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.state = None
        self.players = None

        self.renderer = TronRender(board_size, num_players)

        self.observation_space = Dict({
            'board': Box(0, num_players, shape=(board_size, board_size)),
            'heads': Box(0, np.infty, shape=(num_players,)),
            'directions': Box(0, 4, shape=(num_players,)),
            'deaths': Box(0, num_players, shape=(num_players,))
        })

    def reset(self):
        self.state, self.players = self.env.new_state()
        return {str(i): self.env.state_to_observation(self.state, i) for i in range(self.env.num_players)}

    def step(self, action_dict):
        action_to_string = {
            0: 'forward',
            1: 'right',
            2: 'left'
        }

        actions = []

        for player in self.players:
            action = action_dict.get(str(player), 0)
            actions.append(action_to_string[action])

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)

        num_players = self.env.num_players
        alive_players = set(self.players)

        observations = {str(i): self.env.state_to_observation(self.state, i) for i in map(int, action_dict.keys())}
        rewards = {str(i): rewards[i] for i in map(int, action_dict.keys())}
        dones = {str(i): i not in alive_players for i in map(int, action_dict.keys())}
        dones['__all__'] = terminal

        return observations, rewards, dones, {}

    def render(self, mode='human'):
        if self.state is None:
            return None

        return self.renderer.render(self.state, mode)

    def close(self):
        self.renderer.close()
        
    def test(self, trainer, frame_time = 0.1):
        num_players = self.env.num_players
        self.close()
        state = self.reset()
        done = {"__all__": False}
        action = {str(i): None for i in range(num_players)}
        reward = {str(i): None for i in range(num_players)}
        cumulative_reward = 0
        
        while not done['__all__']:
            action = {i: trainer.compute_action(state[i], prev_action=action[i], prev_reward=reward[i], policy_id=i) for
                      i in map(str, range(num_players))}
            
            state, reward, done, results = self.step(action)
            cumulative_reward += sum(reward.values())
            self.render()
            
            sleep(frame_time)
        
        self.render()
        return cumulative_reward

# Some preprocessing to let the networks learn faster
class TronExtractBoard(Preprocessor):
    def _init_shape(self, obs_space, options):
        board_size = env.observation_space['board'].shape[0]
        return (board_size + 4, board_size + 4, 2)
    
    def transform(self, observation):
        if 'board' in observation:
            return self._transform(observation)
        else:
            return {key: self._transform(value) for key, value in observation.items()}
    
    def _transform(self, observation):
        board = observation['board']
        
        # Make all enemies look the same
        board[board > 1] = -1
        
        # Mark where all of the player heads are
        heads = np.zeros_like(board)
        heads.ravel()[observation['heads']] += 1 + observation['directions']
        
        # Pad the outsides so that we know where the wall is
        board = np.pad(board, 2, 'constant', constant_values=-1)
        heads = np.pad(heads, 2, 'constant', constant_values=-1)
        
        # Combine together
        board = np.expand_dims(board, -1)
        heads = np.expand_dims(heads, -1)
        
        return np.concatenate([board, heads], axis=-1)

# Initialize training environment
ray.init()

def environment_creater(params=None):
    return TronRayEnvironment(board_size=13, num_players=4)

env = environment_creater()
tune.register_env("tron_multi_player", environment_creater)
ModelCatalog.register_custom_preprocessor("tron_prep", TronExtractBoard)

# Configure Deep Q Learning for multi-agent training
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
config["timesteps_per_iteration"] = 128
config['target_network_update_freq'] = 64
config['buffer_size'] = 100_000
config['compress_observations'] = False
config['n_step'] = 2
config["framework"] = "torch"

# Dummy agents
agent_config_dummy = {
    "model": {
        "vf_share_layers": True,
        "conv_filters": [(512, 5, 1), (256, 3, 2), (128, 3, 2), (64, 5, 1)],
        "fcnet_hiddens": [64],
        "custom_preprocessor": 'tron_prep'
    }
}

# Agent with changed params
agent_config_trained = {
    "model": {
        "vf_share_layers": True,
        "conv_filters": [(512, 5, 1), (256, 3, 2), (128, 3, 2), (64, 5, 1)],
        "fcnet_hiddens": [64],
        "custom_preprocessor": 'tron_prep'
    }
}

# index 0 = agent with changed params
config['multiagent'] = {
        "policy_mapping_fn": lambda x, episode, **kwargs: str(x),
        "policies": {str(i): (None, env.observation_space, env.action_space, agent_config_trained if i == 0 else agent_config_dummy) for i in range(env.env.num_players)}
}
       
trainer = DQNTrainer(config, "tron_multi_player")

num_epoch = 100
test_epochs = 1
for epoch in range(num_epoch):
    print("Training iteration: {}".format(epoch), end='')
    res = trainer.train()
    print(f", Average reward: {res['episode_reward_mean']}")
    
    if epoch % test_epochs == 0:
        reward = env.test(trainer)