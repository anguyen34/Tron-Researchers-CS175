# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import random
from copy import deepcopy

import matplotlib.pyplot as plt
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
        self.data_collect_on = False
        self.PLAYER_TRAIN_INDEX = 0
        self.normalize_player_train_wins = False
        self.cumulative_reward_player_train = 0

        self.epsilon = 0.01 # chance of taking a random action instead of the best
        self.actions = ["forward", "left", "right"]

    def reset(self):
        self.state, self.players = self.env.new_state()
        self.cumulative_reward_player_train = 0
        return {str(i): self.env.state_to_observation(self.state, i) for i in range(self.env.num_players)}

    def step(self):
        actions = []

        for player in self.players:
            actions.append(self.choose_action(player))

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)
        if winners is not None and self.PLAYER_TRAIN_INDEX in winners:
            self.normalize_player_train_wins = True
        else:
            self.normalize_player_train_wins = False
        self.cumulative_reward_player_train += (rewards[self.PLAYER_TRAIN_INDEX] if (self.PLAYER_TRAIN_INDEX in self.players) else 0)
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
        
    def test(self, num_epoch, frame_time = 0.1, data_collect_on=False):
        num_players = self.env.num_players
        # init player one's reward list
        self.data_collect_on = data_collect_on
        player_reward_data = []
        for i in range(num_epoch):
            self.close()
            print("Training iteration: {}".format(i))
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
            # Add player one's cumulative reward's to list
            if self.data_collect_on:
                PLAYER_WIN_AMOUNT = 9
                player_reward_data.append(self.cumulative_reward_player_train - (PLAYER_WIN_AMOUNT if self.normalize_player_train_wins else 0))
        # Graph player one's cumulative reward list as Y and iterations 0-99 as X
        if self.data_collect_on:
            plt.plot([i for i in range(0, num_epoch)], player_reward_data)
            plt.xlabel("Iterations (games)")
            plt.ylabel("Reward")
            plt.title("Monte Carlo Cumulative Reward Per Game")
            plt.savefig('docs/images/agent_mcsearch_data_baseline.png', bbox_inches='tight')
        
        self.render()
        return cumulative_reward

    def score(self, players, pno, rewards, winners, state):
        DIED = -50
        WON = 50
        if pno not in players:
            return DIED
        if winners is not None and pno in winners:
            return WON
        board = state[0].flatten()
        return rewards[pno] * len([i for i in board if i == pno])

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
                        return self.score(players_new, pno, rewards, winners, state_new)
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

if __name__ == "__main__":
    agent = MCSearchAgent(depth=2)
    num_epoch = 500
    agent.test(num_epoch, data_collect_on=True)
