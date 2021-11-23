# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import numpy as np
import random
import graphing
from copy import deepcopy
from itertools import combinations_with_replacement

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
import os

from sklearn.ensemble import VotingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

SEED = 12345
np.random.seed(SEED)
env_gbl = None



class MCSearchAgentMA:
    def __init__(self, depth=2, epsilon=0.01, board_size=15, num_players=4, w=50, d=-50):
        self.test_depth = depth
        self.max_depth = 2
        self.epsilon = epsilon # chance of taking a random action instead of the best
        self.actions = ["forward", "left", "right"]
        self.WON = w
        self.DIED = d
        self.PLAYER_TRAIN_INDEX = 1

    def score(self, players, pno, rewards, winners, state):
        if pno not in players:
            return self.DIED
        if winners is not None and pno in winners:
            return self.WON
        board = state[0].flatten()
        return rewards[pno] * len([i for i in board if i == pno])

    def choose_qvals(self, pno, env, state, players):
        moves = {'forward': 0, 'left': 0, 'right': 0}
        for m in moves:
            env_clone = deepcopy(env)
            state_clone = deepcopy(state)
            players_clone = deepcopy(players)
            moves[m] = self.search(state_clone, players_clone, env_clone, m, pno, 0)
        return moves

    def search(self, state, players, env, pmove, pno, depth):
        if pno not in players:
            return self.DIED

        action_combos = combinations_with_replacement(self.actions, len(players) - 1)
        action_states = []
        for i in action_combos:
            c = 0
            astate = []
            for p in players:
                if p == pno:
                    astate.append(pmove)
                else:
                    astate.append(i[c])
                    c += 1
            action_states.append(astate)
        scores = {'forward': None, 'left': None, 'right': None}
        plist = list(players)
        for a in action_states:
            env_clone = deepcopy(env)
            state_clone = deepcopy(state)
            players_clone = deepcopy(players)
            state_new, players_new, rewards, terminal, winners = env_clone.next_state(state_clone, players_clone, a)
            if pno == self.PLAYER_TRAIN_INDEX and depth == self.test_depth:
                return self.score(players_new, pno, rewards, winners, state_new)
            elif pno != self.PLAYER_TRAIN_INDEX and depth == self.max_depth:
                return self.score(players_new, pno, rewards, winners, state_new)
            else:
                new_score = self.search(state_clone, players_clone, env_clone, a[plist.index(pno)], pno, depth + 1)
                if scores[a[plist.index(pno)]] is None:
                    scores[a[plist.index(pno)]] = new_score
                elif new_score > scores[a[plist.index(pno)]]:
                    scores[a[plist.index(pno)]] = new_score
        return scores[pmove]

    def choose_action(self, pno, env, state, players):
        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            act_no = random.randint(0, len(self.actions) - 1)
            return self.actions[act_no]
        else:
            qvals = self.choose_qvals(pno, env, state, players)
            return max(qvals, key=qvals.get)



class RandomForestAgentMA:
    def __init__(self, depth=None, estimators=100, epsilon=0.01, max_leaf=None):
        self.max_depth = depth
        self.est = estimators
        self.rforest = None
        self.train_states = []
        self.train_rewards = []
        self.cumulative_reward = 0
        self.gno = 0
        self.max_leaves = max_leaf

        self.epsilon = epsilon # chance of taking a random action instead of the best
        self.actions = ["forward", "left", "right"]

    def reset(self):
        self.rforest = None
        self.train_states = []
        self.train_rewards = []
        self.cumulative_reward = 0
        self.gno = 0

    def save_data(self, pno, reward, state, move):
        act_to_str = {'forward': 0, 'left': 1, 'right': 2}
        self.cumulative_reward += reward

        b = np.zeros((len(state[0]), len(state[0][0]),))
        for i in range(len(state[0])):
            for j in range(len(state[0][0])):
                if state[0][i][j] == 0:
                    b[i][j] = 0
                elif state[0][i][j] != pno + 1:
                    b[i][j] = -1
                else:
                    b[i][j] = state[0][i][j]
        board = np.pad(b, 2, 'constant', constant_values=-1).flatten()

        act_to_str_np = np.array([act_to_str[move], state[1][pno], state[2][pno], state[3][pno]] + state[1] + state[2] + state[3])
        act_to_str_np = np.append(act_to_str_np, board)
        self.train_states.append(act_to_str_np)
        self.train_rewards.append(self.cumulative_reward)

    def choose_qvals(self, pno, state):
        moves = {'forward': 0, 'left': 0, 'right': 0}
        act_to_str = {'forward': 0, 'left': 1, 'right': 2}
        #Flatten board
        b = np.zeros((len(state[0]), len(state[0][0]),))
        for i in range(len(state[0])):
            for j in range(len(state[0][0])):
                if state[0][i][j] == 0:
                    b[i][j] = 0
                elif state[0][i][j] != pno + 1:
                    b[i][j] = -1
                else:
                    b[i][j] = state[0][i][j]
        board = np.pad(b, 2, 'constant', constant_values=-1).flatten()
        for m in moves:
            # First 3 features to do are planned move, head, direction
            move_to_do = np.array([act_to_str[m], state[1][pno], state[2][pno], state[3][pno]] + state[1] + state[2] + state[3])
            move_to_do = np.append(move_to_do, board)
            move_to_do = move_to_do.reshape(1, -1)
            moves[m] = self.rforests.predict(move_to_do)
        return moves

    def choose_action(self, pno, state):
        # select the next action
        rnd = random.random()
        if self.gno == 0 or rnd < self.epsilon:
            act_no = random.randint(0, len(self.actions) - 1)
            return self.actions[act_no]
        else:
            qvals = self.choose_qvals(pno, state)
            max_qval = max(qvals.values())
            dup_max = [k for k, v in qvals.items() if v == max_qval]
            if len(dup_max) > 1:
                return dup_max[random.randint(0, len(dup_max) - 1)]
            else:
                return max(qvals, key=qvals.get)

    def train(self):
        self.rforests = None
        train_states_np = np.array(self.train_states)
        train_rewards_np = np.array(self.train_rewards)
        rf = RandomForestRegressor(n_estimators=self.est, max_depth=self.max_depth, max_leaf_nodes=self.max_leaves)
        rf = rf.fit(train_states_np, train_rewards_np)
        self.rforests = rf



class EnsembleForestAgentMA:
    def __init__(self, estimators=50, loss='linear', kernel='rbf', activation='relu', hidden_layers=(100,), epsilon=0.01):
        self.est = estimators
        self.ensemble = None
        self.train_states = []
        self.train_rewards = []
        self.cumulative_reward = 0
        self.gno = 0

        self.epsilon = epsilon # chance of taking a random action instead of the best
        self.actions = ["forward", "left", "right"]
        self.est = estimators
        self.loss = loss
        self.kernel = kernel
        self.act = activation
        self.layers=hidden_layers

    def reset(self):
        self.ensemble = None
        self.train_states = []
        self.train_rewards = []
        self.cumulative_reward = 0
        self.gno = 0

    def save_data(self, pno, reward, state, move):
        act_to_str = {'forward': 0, 'left': 1, 'right': 2}
        self.cumulative_reward += reward

        b = np.zeros((len(state[0]), len(state[0][0]),))
        for i in range(len(state[0])):
            for j in range(len(state[0][0])):
                if state[0][i][j] == 0:
                    b[i][j] = 0
                elif state[0][i][j] != pno + 1:
                    b[i][j] = -1
                else:
                    b[i][j] = state[0][i][j]
        board = np.pad(b, 2, 'constant', constant_values=-1).flatten()

        act_to_str_np = np.array([act_to_str[move], state[1][pno], state[2][pno], state[3][pno]] + state[1] + state[2] + state[3])
        act_to_str_np = np.append(act_to_str_np, board)
        self.train_states.append(act_to_str_np)
        self.train_rewards.append(self.cumulative_reward)

    def choose_qvals(self, pno, state):
        moves = {'forward': 0, 'left': 0, 'right': 0}
        act_to_str = {'forward': 0, 'left': 1, 'right': 2}
        #Flatten board
        b = np.zeros((len(state[0]), len(state[0][0]),))
        for i in range(len(state[0])):
            for j in range(len(state[0][0])):
                if state[0][i][j] == 0:
                    b[i][j] = 0
                elif state[0][i][j] != pno + 1:
                    b[i][j] = -1
                else:
                    b[i][j] = state[0][i][j]
        board = np.pad(b, 2, 'constant', constant_values=-1).flatten()
        for m in moves:
            # First 3 features to do are planned move, head, direction
            move_to_do = np.array([act_to_str[m], state[1][pno], state[2][pno], state[3][pno]] + state[1] + state[2] + state[3])
            move_to_do = np.append(move_to_do, board)
            move_to_do = move_to_do.reshape(1, -1)
            moves[m] = self.ensemble.predict(move_to_do)
        return moves

    def choose_action(self, pno, state):
        # select the next action
        rnd = random.random()
        if self.gno == 0 or rnd < self.epsilon:
            act_no = random.randint(0, len(self.actions) - 1)
            return self.actions[act_no]
        else:
            qvals = self.choose_qvals(pno, state)
            max_qval = max(qvals.values())
            dup_max = [k for k, v in qvals.items() if v == max_qval]
            if len(dup_max) > 1:
                return dup_max[random.randint(0, len(dup_max) - 1)]
            else:
                return max(qvals, key=qvals.get)

    def train(self):
        self.ensemble = None
        train_states_np = np.array(self.train_states)
        train_rewards_np = np.array(self.train_rewards)
        ab = AdaBoostRegressor(loss=self.loss, n_estimators=self.est)
        if len(self.train_states) < 4:
            knn = KNeighborsRegressor(n_neighbors=len(self.train_states))
        else:
            knn = KNeighborsRegressor(n_neighbors=4)
        mlp = MLPRegressor(activation=self.act, hidden_layer_sizes=self.layers)
        svr = SVR(kernel=self.kernel)
        vr = VotingRegressor([('ab', ab), ('knn', knn), ('mlp', mlp), ('svr', svr)])
        vr = vr.fit(train_states_np, train_rewards_np)
        self.ensemble = vr



# A full free-for-all version of tron
class TronRayEnvironment(MultiAgentEnv):
    action_space = Discrete(3)

    def __init__(self, board_size=15, num_players=4, epsilon=0.01):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.state = None
        self.players = None
        self.epsilon = epsilon
        self.PLAYER_TRAIN_INDEX = 0
        self.data_collect_on = False
        self.normalize_player_train_wins = False
        self.cumulative_rewards = {}

        self.renderer = TronRender(board_size, num_players)

        self.observation_space = Dict({
            'board': Box(0, num_players, shape=(board_size, board_size)),
            'heads': Box(0, np.infty, shape=(num_players,)),
            'directions': Box(0, 4, shape=(num_players,)),
            'deaths': Box(0, num_players, shape=(num_players,))
        })
        self.mcagent = MCSearchAgentMA(depth=2, epsilon=0.01)
        self.rfagent = RandomForestAgentMA(depth=None, estimators=100, epsilon=0.01, max_leaf=None)
        self.vragent = EnsembleForestAgentMA(estimators=50, loss='linear', kernel='rbf', activation='relu', hidden_layers=(100,), epsilon=0.01)

    def reset(self):
        self.state, self.players = self.env.new_state()
        self.cumulative_rewards = {}
        self.rfagent.reset()
        self.vragent.reset()
        return {str(i): self.env.state_to_observation(self.state, i) for i in range(self.env.num_players)}

    def step(self, action_dict):
        action_to_string = {
            0: 'forward',
            1: 'right',
            2: 'left'
        }

        actions = []
        m2 = None
        m3 = None

        for player in self.players:
            if player == 0:
                action = action_dict.get(str(player), 0)
                rnd = random.random()
                if rnd < self.epsilon:
                    action = random.randint(0, len(action_to_string) - 1)
                actions.append(action_to_string[action])
            elif player == 1:
                actions.append(self.mcagent.choose_action(player, self.env, self.state, self.players))
            elif player == 2:
                actions.append(self.rfagent.choose_action(player, self.state))
                m2 = actions[-1]
            elif player == 3:
                actions.append(self.vragent.choose_action(player, self.state))
                m3 = actions[-1]

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)
        if 2 in self.players:
            self.rfagent.save_data(2, rewards[2], self.state, m2)
        if 3 in self.players:
            self.vragent.save_data(3, rewards[3], self.state, m3)
        if winners is not None and self.PLAYER_TRAIN_INDEX in winners:
            self.normalize_player_train_wins = True
        else:
            self.normalize_player_train_wins = False
        for player in self.players:
            if player not in self.cumulative_rewards:
                self.cumulative_rewards[player] = 0
            self.cumulative_rewards[player] += rewards[player]

        num_players = self.env.num_players
        alive_players = set(self.players)

        observations = {str(i): self.env.state_to_observation(self.state, i) for i in map(int, action_dict.keys())}
        rewards = {str(i): rewards[i] for i in map(int, action_dict.keys())}
        dones = {str(i): i not in alive_players for i in map(int, action_dict.keys())}
        dones['__all__'] = terminal

        return observations, rewards, dones, {}

    def render(self, mode='human'):
        return None
        if self.state is None:
            return None

        return self.renderer.render(self.state, mode)

    def close(self):
        return None
        self.renderer.close()
        
    def test(self, num_epochs, trainer, param=None, val=None, data_collect_on=False, frame_time = 0.1):
        total_rewards = []
        cumulative_reward = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in range(num_epochs):
            print('Training Iteration: {}'.format(i))
            trainer.train()
            if i > 0:
                self.rfagent.train()
                self.vragent.train()
            num_players = self.env.num_players
            self.close()
            state = self.reset()
            done = {"__all__": False}
            action = {str(i): None for i in range(num_players)}
            reward = {str(i): None for i in range(num_players)}
            cumulative_reward = {0: 0, 1: 0, 2: 0, 3: 0}
            
            while not done['__all__']:
                action = {i: trainer.compute_action(state[i], prev_action=action[i], prev_reward=reward[i], policy_id=i) for
                          i in map(str, range(num_players))}
                state, reward, done, results = self.step(action)
                for i in range(len(reward.values())):
                    cumulative_reward[i] += list(reward.values())[i]
                self.render()
                
                sleep(frame_time)
            # Add player one's cumulative reward's to list
            total_rewards.append(cumulative_reward)
        
        self.render()
        # Graph player one's cumulative reward list as Y and iterations 0-99 as X
        ray.shutdown()
        return cumulative_reward

# Some preprocessing to let the networks learn faster
class TronExtractBoard(Preprocessor):
    def _init_shape(self, obs_space, options):
        board_size = env_gbl.observation_space['board'].shape[0]
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

def start_session(num_hidden=1, num_nodes=64, act='relu', epsilon=0.01):
    # Initialize training environment
    ray.init()

    def environment_creater(params=None):
        return TronRayEnvironment(board_size=15, num_players=4, epsilon=epsilon)

    env = environment_creater()
    tune.register_env("tron_multi_player", environment_creater)
    global env_gbl
    env_gbl = env
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
            "fcnet_hiddens": [num_nodes for _ in range(num_hidden)],
            "fcnet_activation": act,
            "custom_preprocessor": 'tron_prep'
        }
    }

    # index 0 = agent with changed params
    config['multiagent'] = {
            "policy_mapping_fn": lambda x, episode, **kwargs: str(x),
            "policies": {str(i): (None, env.observation_space, env.action_space, agent_config_trained if i == 0 else agent_config_dummy) for i in range(env.env.num_players)}
    }
        
    trainer = DQNTrainer(config, "tron_multi_player")
    return trainer, env

num_epoch = 5
trainer, env = start_session()
rewards = env.test(num_epoch, trainer)
print(rewards)
