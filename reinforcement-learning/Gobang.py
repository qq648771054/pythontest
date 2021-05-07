import numpy as np

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
        self.model = self._createModel(self.env.SIZE)
        self.memory = []

    def choose_action(self, state, e_greddy=0.9):
        actions = self.env.validActions(state)
        if np.random.uniform() > e_greddy:
            return np.random.choice(actions)
        else:
            random.shuffle(actions)
            next_states = np.array([self.env.getNextState(state, a) for a in actions])
            next_states = self.addAixs(next_states)
            values = self.model.predict(next_states)
            return actions[values.argmax()]

    def save_exp(self, state, player):
        self.memory.append((state, player))

    def learn(self, winer):
        xs, ys = [], []
        for state, player in self.memory:
            xs.append(state)
            ys.append(1 if player == winer else -1)
        self.model.fit(self.addAixs(np.array(xs)), np.array(ys), epochs=1, verbose=0)
        self.memory = []

    def addAixs(self, data):
        return data.reshape(data.shape + (1, ))

    def clear(self):
        self.memory = []

    def _createModel(self, size, learning_rate=0.01):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(size, size, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(
            loss=tf.losses.mse,
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
        )
        print(model.summary())
        return model

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
        env = envoriment.Gobang(self.agentType)
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
                agent.save_exp(next_state, player)
                step += 1
                state = next_state
                self.render(env, 0.5)
                if winer is not None:
                    break
            if winer == 1:
                win += 1
            elif winer == 2:
                lose += 1
            if win + lose == 100:
                self.winRate.append(win)
                win, lose = 0, 0
            if winer != 0:
                agent.learn(winer)
                agent.clear()
                print('episode {}, winer {}, step {}'.format(episode, winer, step))
                if episode % 10 == 0:
                    self.saveModel(agent)

    def showWinRate(self):
        Image(self.winRate)

if __name__ == '__main__':
    thread = ThreadJing(Agent, showProcess=False, mode=0, savePath=getDataFilePath('GoBang.h5'))
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


