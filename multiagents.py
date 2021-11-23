# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import numpy as np
import random
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

from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models import ModelCatalog

from sklearn.ensemble import VotingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

SEED = 12345
np.random.seed(SEED)



class MCSearchAgentMA:
    def __init__(self, depth=2, epsilon=0.01, board_size=15, num_players=4, w=50, d=-50):
        self.test_depth = depth
        self.max_depth = 2
        self.epsilon = epsilon # chance of taking a random action instead of the best
        self.actions = ["forward", "left", "right"]
        self.WON = w
        self.DIED = d

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

    def save_data(self, pno, reward, state):
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

        act_to_str_np = np.array([act_to_str[pno], state[1][pno], state[2][pno], state[3][pno]] + state[1] + state[2] + state[3])
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

    def save_data(self, pno, reward, state):
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

        act_to_str_np = np.array([act_to_str[pno], state[1][pno], state[2][pno], state[3][pno]] + state[1] + state[2] + state[3])
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



# A version of tron where only one agent may learn and the others are fixed
class TronRayMultipleAgents(gym.Env):
    def __init__(self, board_size=15, num_players=4, spawn_offset=2):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.state = None
        self.players = None
        self.human_player = None
        self.spawn_offset = spawn_offset

        self.renderer = TronRender(board_size, num_players, winner_player=0)
        
        self.action_space = Discrete(3)
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
        self.state, self.players = self.env.new_state(spawn_offset=self.spawn_offset)
        self.human_player = 0
        self.rfagent.reset()
        self.vragent.reset()
        return self._get_observation(self.human_player)

    def _get_observation(self, player):
        return self.env.state_to_observation(self.state, player)

    def step(self, action: int):
        human_player = self.human_player

        action_to_string = {
            0: 'forward',
            1: 'right',
            2: 'left'
        }

        actions = []
        for player in self.players:
            if player == human_player:
                actions.append(action_to_string[action])
            elif player == 1:
                actions.append(self.mcagent.choose_action(player, self.env, self.state, self.players))
            elif player == 2:
                actions.append(self.rfagent.choose_action(player, self.state))
            elif player == 3:
                actions.append(self.vragent.choose_action(player, self.state))

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)

        if 2 in self.players:
            self.rfagent.save_data(2, rewards[2], self.state)
        if 3 in self.players:
            self.rfagent.save_data(3, rewards[3], self.state)
        observation = self._get_observation(human_player)
        reward = rewards[human_player]
        done = terminal
        return observation, reward, done, {}

    def render(self, mode='human'):
        if self.state is None:
            return None
        return self.renderer.render(self.state, mode)

    def close(self):
        self.renderer.close()
        
    def test(self, trainer, frame_time = 0.1, epochs = 100):
        self.close()
        state = self.reset()
        done = False
        action = None
        reward = None
        cumulative_reward = 0
        for i in range(epochs):
            print('Training Iteration: {}'.format(i))
            self.rfagent.train()
            self.vragent.train()
            trainer.train()
            while not done:
                action = trainer.compute_action(state, prev_action=action, prev_reward=reward)
                state, reward, done, results = self.step(action)
                cumulative_reward += reward
                self.render()
                sleep(frame_time)
            self.render()
        return cumulative_reward



# Some preprocessing to let the network learn faster
class TronExtractBoard(Preprocessor):
    def _init_shape(self, obs_space, options):
        board_size = env.observation_space['board'].shape[0]
        return (board_size + 4, board_size + 4, 2)

    def transform(self, observation):
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
    return TronRaySinglePlayerEnvironment(board_size=15, num_players=4)

env = environment_creater()
tune.register_env("tron_agent_battle", environment_creater)
ModelCatalog.register_custom_preprocessor("tron_prep", TronExtractBoard)

# Configure Deep Q Learning with reasonable values
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 4
config['num_gpus'] = 1
config["timesteps_per_iteration"] = 1024
config['target_network_update_freq'] = 128
config['buffer_size'] = 1_000_000
config['compress_observations'] = False
config['n_step'] = 3
config['seed'] = SEED
config["framework"] = "torch"

# We will use a simple convolutiotrainer.save("./checkpoint")n network with 3 layers as our feature extractor
config['model']['vf_share_layers'] = True
config['model']['conv_filters'] = [(512, 5, 1), (256, 3, 2), (128, 3, 2), (64, 5, 1)]
config['model']['fcnet_hiddens'] = [64]
config['model']['custom_preprocessor'] = 'tron_prep'

# Begin training or evaluation
trainer = DQNTrainer(config, "tron_agent_battle")

num_epoch = 100
reward = env.test(trainer, epochs=num_epoch)