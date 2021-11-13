# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import random
from copy import deepcopy

from matplotlib import cm
from time import sleep
from colosseumrl.envs.tron import TronGridEnvironment, TronRender
from colosseumrl.envs.tron.rllib import TronRaySinglePlayerEnvironment
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
os.environ['SDL_VIDEODRIVER']='x11'

# Features: Previous move, current board state, actual reward
# Predict reward based on next move and current board state
class MCForestAgent:
    def __init__(self, depth, board_size=15, num_players=4, estimators=50):
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
        self.gno = 0

        self.epsilon = 0.01 # chance of taking a random action instead of the best
        self.actions = ["forward", "left", "right"]

    def reset(self):
        self.state, self.players = self.env.new_state()
        return {str(i): self.env.state_to_observation(self.state, i) for i in range(self.env.num_players)}

    def step(self):
        act_to_str = {'forward': 0, 'left': 1, 'right': 2}
        actions = []

        for player in self.players:
            actions.append(self.choose_action(player))

        old_players = deepcopy(self.players)
        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)

        # Saving additional data for games.
        b = self.state[0].flatten()
        for r in range(len(old_players)):
            act_to_str_np = np.array([act_to_str[actions[r]], self.state[1][old_players[r]], self.state[2][old_players[r]]])
            act_to_str_np = np.append(act_to_str_np, b)


            self.train_states[old_players[r]].append(act_to_str_np)
            self.train_rewards[old_players[r]].append(rewards[old_players[r]])

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
        
    def test(self, frame_time = 0.1):
        total_rewards = []
        num_players = self.env.num_players
        self.close()
        for i in range(5):
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
                self.render()
                
                sleep(frame_time)
            
            self.render()
            total_rewards.append(cumulative_reward)
            self.gno += 1
        return total_rewards

    def choose_qvals(self, pno):
        moves = {'forward': 0, 'left': 0, 'right': 0}
        act_to_str = {'forward': 0, 'left': 1, 'right': 2}
        #Flatten board
        b = self.state[0].flatten()
        for m in moves:
            # First 3 features to do are planned move, head, direction
            move_to_do = np.array([act_to_str[m], self.state[1][pno], self.state[2][pno]])
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
            rf = RandomForestRegressor(self.est)
            #train_states_np.reshape(1, -1)
            rf.fit(train_states_np, train_rewards_np)
            self.rforests.append(rf)



agent = MCForestAgent(depth=10)
num_epoch = 100
test_epochs = 1
for epoch in range(num_epoch):
    print("Training iteration: {}".format(epoch))
    agent.test()
