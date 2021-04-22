from lib import *
import matplotlib.animation as animation
import time
import tkinter as tk
import random
import Thread
import envoriment
import math

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.model = np.zeros((self.env.STATE_SIZE, self.env.ACTION_SIZE))
        self.memory = []

    def choose_action(self, state, e_greddy=0.9):
        validActions = self.env.validActions(state)
        if np.random.uniform() > e_greddy:
            return np.random.choice(validActions)
        else:
            return self.filter_action(state, validActions, lambda a, b: a > b)

    def filter_action(self, state, actions, func):
        random.shuffle(actions)
        res = actions[0]
        for i in actions:
            if func(self.model[state, i], self.model[state, res]):
                res = i
        return res

    def save_exp(self, state, action, next_state, player):
        self.memory.append((state, action, next_state, player))

    def learn(self, winer, learning_rate=0.01):
        for state, action, next_state, player in self.memory:
            q_predict = self.model[state, action]
            q_target = 1 if player == winer else -1
            self.model[state, action] += learning_rate * (q_target - q_predict)
        self.memory = []

    def clear(self):
        self.memory = []

class Image(object):
    def __init__(self, data):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.data = data
        ani = animation.FuncAnimation(self.fig, self.run, blit=True, interval=1000,
                                      repeat=False, init_func=self.init)
        plt.show()

    def getData(self):
        l = len(self.data)
        start = 0 if l < 100 else l - 100
        x = [i for i in range(start, l)]
        y = [self.data[i] for i in range(start, l)]
        return x, y

    def init(self):
        x, y = self.getData()
        self.line.set_data(x, y)
        self.ax.set_ylim(0, 1000)
        self.ax.set_xlim(x[0], x[len(x) - 1])
        self.ax.figure.canvas.draw()
        return self.line,

    def run(self, data):
        x, y = self.getData()
        self.line.set_data(x, y)
        self.ax.set_ylim(0, 1000)
        self.ax.set_xlim(x[0], x[len(x) - 1])
        self.ax.figure.canvas.draw()
        return self.line,

class ThreadJing(Thread.ThreadBase):
    def run(self):
        env = envoriment.Jing(self.agentType)
        agent = env.agent
        self.loadModel(agent)
        episode = 0
        self.winRate = []
        win, lose = 0, 0
        while True:
            state = env.reset()
            self.render(env, 0.5)
            episode += 1
            step = 0
            while True:
                action = agent.choose_action(state, 0.9 if self.mode == 0 else 1.0)
                next_state, player, winer = env.step(action)
                agent.save_exp(state, action, next_state, player)
                step += 1
                state = next_state
                self.render(env, 0.5)
                if winer is not None:
                    break
            if winer == 1:
                win += 1
            elif winer == 2:
                lose += 1
            if win + lose == 1000:
                self.winRate.append(win)
                win, lose = 0, 0
            if winer != 0:
                agent.learn(winer)
                agent.clear()
                # print('\repisode {}, winer {}, step {}'.format(episode, winer, step), end='')
            if episode % 1000 == 0:
                self.saveModel(agent)

    def showWinRate(self):
        Image(self.winRate)

    def loadModel(self, agent):
        if self.savePath and os.path.exists(self.savePath):
            agent.model = np.load(self.savePath)

    def saveModel(self, agent):
        if self.savePath:
            np.save(self.savePath, agent.model)

if __name__ == '__main__':
    thread = ThreadJing(Agent, showProcess=False, mode=1)
    thread.start()
    while True:
        cmd = input()
        if cmd == 'process 0':
            thread.showProcess = False
        elif cmd == 'process 1':
            thread.showProcess = True
        elif cmd == 'image':
            thread.showWinRate()
        elif cmd == 'mode 0':
            thread.mode = 0
        elif cmd == 'mode 1':
            thread.mode = 1


