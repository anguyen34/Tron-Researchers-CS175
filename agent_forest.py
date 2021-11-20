# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import random
from copy import deepcopy

import matplotlib.pyplot as plt
from time import sleep
from colosseumrl.envs.tron import TronGridEnvironment, TronRender
from colosseumrl.envs.tron.rllib import TronRaySinglePlayerEnvironment
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
os.environ['SDL_VIDEODRIVER']='x11'
import graphing

# Features: Previous move, current board state, actual reward
# Predict reward based on next move and current board state
class RandomForestAgent:
    def __init__(self, depth=None, board_size=15, num_players=4, estimators=50, epsilon=0.01, max_leaf=None):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.num_players = num_players
        self.state = None
        self.players = None
        self.max_depth = depth
        self.est = estimators
        self.test_depth = None
        self.renderer = TronRender(board_size, num_players)
        self.rforests = []
        self.train_states = [[] for _ in range(num_players)]
        self.train_rewards = [[] for _ in range(num_players)]
        self.cumulative_rewards = {}
        self.gno = 0
        self.data_collect_on = False
        self.PLAYER_TRAIN_INDEX = 0
        self.normalize_player_train_wins = False
        self.max_leaves = max_leaf

        self.epsilon = epsilon # chance of taking a random action instead of the best
        self.actions = ["forward", "left", "right"]

    def reset(self):
        self.state, self.players = self.env.new_state()
        self.cumulative_rewards = {}
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
        b = self.state[0].flatten()
        for r in range(len(old_players)):
            act_to_str_np = np.array([act_to_str[actions[r]], self.state[1][old_players[r]], self.state[2][old_players[r]], self.state[3][old_players[r]]])
            act_to_str_np = np.append(act_to_str_np, b)


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
                cumulative_reward += sum(reward.values())
                #self.render()
                sleep(frame_time)
            # Add player one's cumulative reward's to list
            if self.data_collect_on:
                PLAYER_WIN_AMOUNT = 9
                player_reward_data.append(self.cumulative_reward_player_train - (PLAYER_WIN_AMOUNT if self.normalize_player_train_wins else 0))
                player_delta_data.append(self.cumulative_rewards[self.PLAYER_TRAIN_INDEX] - (PLAYER_WIN_AMOUNT if self.normalize_player_train_wins else 0) - np.average([v for k, v in self.cumulative_rewards.items() if k != self.PLAYER_TRAIN_INDEX]))
            #self.render()
            total_rewards.append(cumulative_reward)
            self.gno += 1
        # Graph player one's cumulative reward list as Y and iterations 0-99 as X
        if self.data_collect_on:
            graphing.plot_graph(num_epoch, player_reward_data, 'Iterations (Num Games/Epochs)', 'Cumulative Reward', 'Cumulative Reward of Altered Agent', 'forest_cumulative_{}_{}.png'.format(param, val))
            graphing.plot_graph(num_epoch, player_delta_data, 'Iterations (Num Games/Epochs)', 'Delta Reward', 'Difference in Rewards of Altered Agent vs. Avg of Others', 'forest_delta_{}_{}.png'.format(param, val))
        return total_rewards

    def choose_qvals(self, pno):
        moves = {'forward': 0, 'left': 0, 'right': 0}
        act_to_str = {'forward': 0, 'left': 1, 'right': 2}
        #Flatten board
        b = self.state[0].flatten()
        for m in moves:
            # First 3 features to do are planned move, head, direction
            move_to_do = np.array([act_to_str[m], self.state[1][pno], self.state[2][pno], self.state[3][pno]])
            move_to_do = np.append(move_to_do, b)
            move_to_do = move_to_do.reshape(1, -1)
            moves[m] = self.rforests[pno].predict(move_to_do)
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
        self.rforests = []
        for i in range(len(self.players)):
            train_states_np = np.array(self.train_states[i])
            train_rewards_np = np.array(self.train_rewards[i])
            rf = RandomForestRegressor(n_estimators=self.est, max_depth=self.max_depth, max_leaf_nodes=self.max_leaves)
            rf = rf.fit(train_states_np, train_rewards_np)
            self.rforests.append(rf)


if __name__ == "__main__":
    num_epoch = 200
    epochs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Num Estimators
    data_estimators = []
    num_estimators = []
    for ne in num_estimators:
        agent = RandomForestAgent(estimators=ne)
        rewards = agent.test(num_epoch, 'estimators', ne, data_collect_on=True)
        data_estimators.append([rewards[19], rewards[39], rewards[59], rewards[79], rewards[99], rewards[119], rewards[139], rewards[159], rewards[179], rewards[199]])
    graphing.plot_heatmap(num_estimators, epochs, data_estimators, 'Num Estimators', 'Epoch', 'Forest Agent Num Estimators (Cumulative Reward)', 'forest_heat_estimators.png')

    # Max Depth
    data_depth = []
    max_depth = []
    for md in max_depth:
        agent = RandomForestAgent(depth=md)
        rewards = agent.test(num_epoch, 'Max Depth', md, data_collect_on=True)
        data_estimators.append([rewards[19], rewards[39], rewards[59], rewards[79], rewards[99], rewards[119], rewards[139], rewards[159], rewards[179], rewards[199]])
    graphing.plot_heatmap(num_estimators, epochs, data_depth, 'Max Depth', 'Epoch', 'Forest Agent Max Depth (Cumulative Reward)', 'forest_heat_maxdepth.png')

    # Max Leaf Nodes
    data_leaves = []
    leaf_nodes = []
    for ln in leaf_nodes:
        agent = RandomForestAgent(max_leaf=ln)
        rewards = agent.test(num_epoch, 'Max Leaf Nodes', ln, data_collect_on=True)
        data_estimators.append([rewards[19], rewards[39], rewards[59], rewards[79], rewards[99], rewards[119], rewards[139], rewards[159], rewards[179], rewards[199]])
    graphing.plot_heatmap(num_estimators, epochs, data_depth, 'Max Leaf Nodes', 'Epoch', 'Forest Agent Max Leaf Nodes (Cumulative Reward)', 'forest_heat_leafnodes.png')

    # Epsilon
    data_epsilon = []
    epsilon = []
    for ep in epsilon:
        agent = RandomForestAgent(epsilon=ep)
        rewards = agent.test(num_epoch, 'Epsilon', ep, data_collect_on=True)
        data_estimators.append([rewards[19], rewards[39], rewards[59], rewards[79], rewards[99], rewards[119], rewards[139], rewards[159], rewards[179], rewards[199]])
    graphing.plot_heatmap(num_estimators, epochs, data_depth, 'Epsilon', 'Epoch', 'Forest Agent Epsilon (Cumulative Reward)', 'forest_heat_epsilon.png')
