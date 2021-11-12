# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import random
from copy import deepcopy

from matplotlib import cm
from time import sleep
from colosseumrl.envs.tron import TronGridEnvironment, TronRender
from colosseumrl.envs.tron.rllib import TronRaySinglePlayerEnvironment


class MCSearchAgent:
    def __init__(self, depth, board_size=15, num_players=4):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.state = None
        self.players = None
        self.max_depth = depth
        self.test_depth = None
        self.renderer = TronRender(board_size, num_players)

        self.epsilon = 0.01 # chance of taking a random action instead of the best
        self.actions = ["forward", "left", "right"]

    def reset(self):
        self.state, self.players = self.env.new_state()
        return {str(i): self.env.state_to_observation(self.state, i) for i in range(self.env.num_players)}

    def step(self):
        actions = []

        for player in self.players:
            actions.append(self.choose_action(self.prev_r, self.prev_s, player))

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)
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
        
    def test(self, trainer, frame_time = 0.1):
        num_players = self.env.num_players
        self.close()
        state = self.reset()
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
        return cumulative_reward

    def score(players, pno, rewards, terminal, winners):
        pass

    def choose_qvals(self, pno):
        moves = {'forward': 0, 'left': 0, 'right': 0}
        for m in moves:
            env_clone = deepcopy(self.env)
            state_clone = deepcopy(self.state)
            players_clone = deepcopy(self.players)
            moves[m] = self.search(state_clone, players_clone, env_clone, m, pno, 0)
        return moves

    def search(self, state, players, env, pmove, pno, depth):
        actions = [None] * len(self.players)
        actions[pno] = pmove
        p = pno + 1
        for m1 in self.actions:
            actions[p % len(self.players)] = m1
            p += 1
            for m2 in self.actions:
                actions[p % len(self.players)] = m2
                p += 1
                for m3 in self.actions:
                    actions[p % len(self.players)] = m3
                    state_new, players_new, rewards, terminal, winners = env.next_state(state, players, actions)
                    if depth == self.max_depth:
                        #return self.score(players_new, pno, rewards, terminal, winners)
                        return rewards[pno]
                    else:
                        scores = {'forward': 0, 'left': 0, 'right': 0}
                        for m in scores:
                            env_clone = deepcopy(env)
                            state_clone = deepcopy(state_new)
                            players_clone = deepcopy(players_new)
                            scores[m] = self.search(state_clone, players_clone, env_clone, m, pno, depth + 1)
                        return max(scores.values())

    def choose_action(self, pno):
        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            act_no = random.randint(0, len(self.actions) - 1)
            return self.actions[act_no]
        else:
            qvals = self.choose_qvals(pno)
            return max(qvals, key=qvals.get)

agent = MCSearchAgent
num_epoch = 100
test_epochs = 1
for epoch in range(num_epoch):
    print("Training iteration: {}".format(epoch), end='')
    MCSearchAgent
