from lib import *
import time
import tkinter as tk
import random
import thread
import envoriment

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((self.env.stateSize, self.env.actionSize))
        self.memory = []

    def choose_action(self, state, e_greddy=0.9):
        validActions = self.env.validActions(state)
        if np.random.uniform() >= e_greddy:
            return np.random.choice(validActions)
        else:
            return self.filter_action(state, validActions, lambda a, b: a > b)

    def filter_action(self, state, actions, func):
        actions.sort(lambda a, b: random.random() - 0.5)
        res = 0
        for i in actions:
            if func(self.q_table[state, i], self.q_table[state, res]):
                res = i
        return res

    def save_exp(self, state, action, next_state, player):
        self.memory.append((state, action, next_state, player))

    def learn(self, winer, learning_rate=0.01):
        for state, action, next_state, player in self.memory:
            reward = 1 if player == winer else -1
            q_predict = self.q_table[state, action]
            q_target = reward + self.q_table[next_state, self.filter_action(next_state, self.env.validActions(next_state), lambda a, b: a < b)]
            self.q_table[state, action] += learning_rate * (q_target - q_predict)
        self.memory = []

class ThreadJing(thread.ThreadBase):
    def run(self):
        env = envoriment.Jing(self.agentType)
        agent = env.agent
        self.loadModel(agent)
        episode = 0
        while True:
            state = env.reset()
            self.render(env, 0.5)
            episode += 1
            step = 0
            while True:
                action = agent.choose_action(state)
                next_state, player, done = env.step(action)
                agent.save_exp(state, action, next_state, player)
                step += 1
                state = next_state
                self.render(env)
                if done:
                    break
            agent.learn(player)
            print('episode {}, winer {}, step {}'.format(episode, player, step))
            self.saveModel(agent)

if __name__ == '__main__':
    thread = ThreadJing(Agent, showProcess=True)
    thread.start()
    while True:
        cmd = input()
        if cmd == '0':
            thread.showProcess = False
        else:
            thread.showProcess = True

