# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import random
from copy import deepcopy

from time import sleep
from colosseumrl.envs.tron import TronGridEnvironment, TronRender
from colosseumrl.envs.tron.rllib import TronRaySinglePlayerEnvironment
from sklearn.ensemble import VotingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import os
import graphing
os.environ['SDL_VIDEODRIVER']='x11'

# Features: Previous move, current board state, actual reward
# Predict reward based on next move and current board state
class EnsembleAgent:
    def __init__(self, board_size=15, num_players=4, estimators=50, loss='linear', kernel='rbf', activation='relu', hidden_layers=(100,), epsilon=0.01):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.num_players = num_players
        self.state = None
        self.players = None
        self.renderer = TronRender(board_size, num_players)
        self.rEnsembles = []
        self.train_states = [[] for _ in range(num_players)]
        self.train_rewards = [[] for _ in range(num_players)]
        self.cumulative_rewards = {}
        self.gno = 0
        self.data_collect_on = False
        self.PLAYER_TRAIN_INDEX = 0
        self.normalize_player_train_wins = False
        self.cumulative_reward_player_train = 0

        self.epsilon = epsilon # chance of taking a random action instead of the best
        self.actions = ["forward", "left", "right"]

        self.est = estimators
        self.loss = loss
        self.kernel = kernel
        self.act = activation
        self.layers = hidden_layers

    def reset(self):
        self.state, self.players = self.env.new_state()
        self.cumulative_rewards = {}
        self.cumulative_reward_player_train = 0
        self.normalize_player_train_wins = False
        return {str(i): self.env.state_to_observation(self.state, i) for i in range(self.env.num_players)}

    def step(self):
        act_to_str = {'forward': 0, 'left': 1, 'right': 2}
        actions = []

        for player in self.players:
            actions.append(self.choose_action(player))

        old_players = deepcopy(self.players)
        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)
        if winners is not None and self.PLAYER_TRAIN_INDEX in winners:
            self.normalize_player_train_wins = True
        else:
            self.normalize_player_train_wins = False

        for player in self.players:
            if player not in self.cumulative_rewards:
                self.cumulative_rewards[player] = 0
            self.cumulative_rewards[player] += rewards[player]

        # Saving additional data for games.
        for r in range(len(old_players)):
            b = np.zeros((len(self.state[0]), len(self.state[0][0]),))
            for i in range(len(self.state[0])):
                for j in range(len(self.state[0][0])):
                    if self.state[0][i][j] == 0:
                        b[i][j] = 0
                    elif self.state[0][i][j] != old_players[r] + 1:
                        b[i][j] = -1
                    else:
                        b[i][j] = self.state[0][i][j]
            board = np.pad(b, 2, 'constant', constant_values=-1).flatten()
            act_to_str_np = np.array([act_to_str[actions[r]], self.state[1][old_players[r]], self.state[2][old_players[r]], self.state[3][old_players[r]]] + self.state[1] + self.state[2] + self.state[3])
            act_to_str_np = np.append(act_to_str_np, board)
            self.train_states[old_players[r]].append(act_to_str_np)
            self.train_rewards[old_players[r]].append(self.cumulative_rewards[old_players[r]])

        num_players = self.env.num_players
        alive_players = set(self.players)

        observations = {str(i): self.env.state_to_observation(self.state, i) for i in range(4)}
        rewards = {str(i): rewards[i] for i in range(4)}
        dones = {str(i): i not in alive_players for i in range(4)}
        dones['__all__'] = terminal

        return observations, rewards, dones, {}

    def render(self, mode='human'):
        if self.state is None:
            return None
        return self.renderer.render(self.state, mode)

    def close(self):
        self.renderer.close()
        
    def test(self, num_epoch, param, val, frame_time = 0.1, data_collect_on = False):
        total_rewards = []
        num_players = self.env.num_players
        # init player one's reward list
        self.data_collect_on = data_collect_on
        player_reward_data = []
        player_delta_data = []
        for i in range(num_epoch):
            #self.close()
            print("Training iteration: {}".format(i))
            state = self.reset()
            if i > 0:
                self.train()
            done = {"__all__": False}
            action = {str(i): None for i in range(num_players)}
            reward = {str(i): None for i in range(num_players)}
            cumulative_reward = 0
            
            while not done['__all__']:
                state, reward, done, results = self.step()
                cumulative_reward += list(reward.values())[self.PLAYER_TRAIN_INDEX]
                #self.render()
                sleep(frame_time)

            # Add player one's cumulative reward's to list
            if self.data_collect_on:
                PLAYER_WIN_AMOUNT = 9
                player_reward_data.append(self.cumulative_rewards[self.PLAYER_TRAIN_INDEX] - (PLAYER_WIN_AMOUNT if self.normalize_player_train_wins else 0))
                player_delta_data.append(self.cumulative_rewards[self.PLAYER_TRAIN_INDEX] - (PLAYER_WIN_AMOUNT if self.normalize_player_train_wins else 0) - np.average([v for k, v in self.cumulative_rewards.items() if k != self.PLAYER_TRAIN_INDEX]))
            #self.render()
            total_rewards.append(cumulative_reward)
            self.gno += 1
        # Graph player one's cumulative reward list as Y and iterations 0-99 as X
        if self.data_collect_on:
            graphing.plot_graph(num_epoch, player_reward_data, 'Iterations (Num Games/Epochs)', 'Cumulative Reward', 'Cumulative Reward of Altered Ensemble Agent', 'ensemble_cumulative_{}_{}.png'.format(param, val))
            graphing.plot_graph(num_epoch, player_delta_data, 'Iterations (Num Games/Epochs)', 'Delta Reward', 'Difference in Rewards of Altered Ensemble Agent vs. Avg of Others', 'ensemble_delta_{}_{}.png'.format(param, val))
        return total_rewards

    def choose_qvals(self, pno):
        moves = {'forward': 0, 'left': 0, 'right': 0}
        act_to_str = {'forward': 0, 'left': 1, 'right': 2}
        #Flatten board
        b = np.zeros((len(self.state[0]), len(self.state[0][0]),))
        for i in range(len(self.state[0])):
            for j in range(len(self.state[0][0])):
                if self.state[0][i][j] == 0:
                    b[i][j] = 0
                elif self.state[0][i][j] != pno + 1:
                    b[i][j] = -1
                else:
                    b[i][j] = self.state[0][i][j]
        board = np.pad(b, 2, 'constant', constant_values=-1).flatten()
        for m in moves:
            # First 3 features to do are planned move, head, direction
            move_to_do = np.array([act_to_str[m], self.state[1][pno], self.state[2][pno], self.state[3][pno]] + self.state[1] + self.state[2] + self.state[3])
            move_to_do = np.append(move_to_do, board)
            move_to_do = move_to_do.reshape(1, -1)
            moves[m] = self.rEnsembles[pno].predict(move_to_do)
        return moves

    def choose_action(self, pno):
        # select the next action
        rnd = random.random()
        if self.gno == 0:
            act_no = random.randint(0, len(self.actions) - 1)
            return self.actions[act_no]
        elif rnd < self.epsilon:
            act_no = random.randint(0, len(self.actions) - 1)
            return self.actions[act_no]
        else:
            qvals = self.choose_qvals(pno)
            max_qval = max(qvals.values())
            dup_max = [k for k, v in qvals.items() if v == max_qval]
            if len(dup_max) > 1:
                return dup_max[random.randint(0, len(dup_max) - 1)]
            else:
                return max(qvals, key=qvals.get)

    def train(self):
        # Voting Ensemble containing: SVR SVM, Adaboost, Nearest Neightbors, MLPRegressor (neural network)
        self.rEnsembles = []
        for i in range(len(self.players)):
            train_states_np = np.array(self.train_states[i])
            train_rewards_np = np.array(self.train_rewards[i])
            ab = AdaBoostRegressor(loss=self.loss, n_estimators=self.est)
            if len(self.train_states[i]) < 4:
                knn = KNeighborsRegressor(n_neighbors=len(self.train_states[i]))
            else:
                knn = KNeighborsRegressor(n_neighbors=4)
            mlp = MLPRegressor(activation=self.act, hidden_layer_sizes=self.layers)
            svr = SVR(kernel=self.kernel)
            vr = VotingRegressor([('ab', ab), ('knn', knn), ('mlp', mlp), ('svr', svr)])
            vr = vr.fit(train_states_np, train_rewards_np)
            self.rEnsembles.append(vr)



if __name__ == "__main__":
    num_epoch = 100
    epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # # AdaBoost Num Estimators
    # data_estimators = []
    # num_estimators = [10, 25, 50, 100, 150, 200]
    # for ne in num_estimators:
    #     agent = EnsembleAgent(estimators=ne)
    #     rewards = agent.test(num_epoch, 'ada_estimators', ne, data_collect_on=True)
    #     data_estimators.append([rewards[9], rewards[19], rewards[29], rewards[39], rewards[49], rewards[59], rewards[69], rewards[79], rewards[89], rewards[99]])
    # graphing.plot_heatmap(num_estimators, epochs, data_estimators, 'Num Estimators', 'Epoch', 'Ensemble Agent AdaBoost Num Estimators', 'densemble_heat_ada_estimators.png')

    # # AdaBoost Loss Function
    # data_functions = []
    # loss_functions = ['linear', 'square', 'exponential']
    # for lf in loss_functions:
    #     agent = EnsembleAgent(loss=lf)
    #     rewards = agent.test(num_epoch, 'ada_loss', lf, data_collect_on=True)
    #     data_functions.append([rewards[9], rewards[19], rewards[29], rewards[39], rewards[49], rewards[59], rewards[69], rewards[79], rewards[89], rewards[99]])
    # graphing.plot_heatmap(loss_functions, epochs, data_functions, 'Loss Function', 'Epoch', 'Ensemble Agent AdaBoost Loss Function', 'ensemble_heat_ada_lossfunction.png')

    # # Support Vector Kernal
    # data_kernel = []
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    # for kr in kernels:
    #     agent = EnsembleAgent(kernel=kr)
    #     rewards = agent.test(num_epoch, 'svr_kernel', kr, data_collect_on=True)
    #     data_kernel.append([rewards[9], rewards[19], rewards[29], rewards[39], rewards[49], rewards[59], rewards[69], rewards[79], rewards[89], rewards[99]])
    # graphing.plot_heatmap(kernels, epochs, data_kernel, 'Kernel', 'Epoch', 'Ensemble Agent Support Vector Regressor Kernal', 'ensemble_heat_svr_kernal.png')

    # Multilayer Perceptron Activation
    data_activation = []
    activations = ['identity', 'logistic', 'relu', 'tanh']
    for av in activations:
        agent = EnsembleAgent(activation=av)
        rewards = agent.test(num_epoch, 'mlp_activation', av, data_collect_on=True)
        data_activation.append([rewards[9], rewards[19], rewards[29], rewards[39], rewards[49], rewards[59], rewards[69], rewards[79], rewards[89], rewards[99]])
    graphing.plot_heatmap(activations, epochs, data_activation, 'Activation Function', 'Epoch', 'Ensemble Agent Multi-layer Perceptron Activation Function', 'ensemble_heat_mlp_activation.png')

    # Multilayer Perceptron Num Hidden Layers
    data_numlayers = []
    num_layers = [1, 2, 5, 10, 25, 50, 100, 200]
    for nl in num_layers:
        agent = EnsembleAgent(hidden_layers=(100) * num_layers)
        rewards = agent.test(num_epoch, 'mlp_numlayers', nl, data_collect_on=True)
        data_numlayers.append([rewards[9], rewards[19], rewards[29], rewards[39], rewards[49], rewards[59], rewards[69], rewards[79], rewards[89], rewards[99]])
    graphing.plot_heatmap(num_layers, epochs, data_numlayers, 'Num Hidden Layers', 'Epoch', 'Ensemble Agent Multi-layer Perceptron Num Hidden Layers', 'ensemble_heat_mlp_numlayers.png')

    # Epsilon
    data_epsilon = []
    epsilon = [0.01, 0.05, 0.1, 0.25, 0.5]
    for ep in epsilon:
        agent = EnsembleAgent(epsilon=ep)
        rewards = agent.test(num_epoch, 'Epsilon', ep, data_collect_on=True)
        data_epsilon.append([rewards[19], rewards[39], rewards[59], rewards[79], rewards[99], rewards[119], rewards[139], rewards[159], rewards[179], rewards[199]])
    graphing.plot_heatmap(epsilon, epochs, data_epsilon, 'Epsilon', 'Epoch', 'Ensemble Agent Epsilon', 'ensemble_heat_epsilon.png')
