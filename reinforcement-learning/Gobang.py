import json

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
        self.historySize = 5000
        self.historyIter = 0
        self.history = []

    def choose_action(self, state, e_greddy=0.9):
        actions = self.env.validActions(state)
        if np.random.uniform() > e_greddy:
            return np.random.choice(actions)
        else:
            values = self.model.predict(np.array([self.addAixs(state)]))[0]
            maxi, maxv = actions[0], values[actions[0]]
            for i, v in enumerate(values):
                if i in actions and v > maxv:
                    maxi, maxv = i, values[i]
            return maxi

    def save_exp(self, state, action, player):
        self.memory.append((self.addAixs(state), action, player))

    def save_history(self, state, action, value):
        if len(self.history) < self.historySize:
            self.history.append((state, action, value))
        else:
            self.history[self.historyIter] = (state, action, value)
            self.historyIter = (self.historyIter + 1) % self.historySize

    def sample(self, batch=128):
        if len(self.history) < batch:
            return (np.array([h[i] for h in self.history]) for i in range(3))
        else:
            idx = np.random.choice(len(self.history), size=batch)
            return (np.array([self.history[x][i] for x in idx]) for i in range(3))

    def learn(self):
        states, actions, values = self.sample(1024)
        predict = self.model.predict(states)
        for p, a, v in zip(predict, actions, values):
            p[a] = v
        self.model.fit(states, predict, epochs=1, verbose=0)

    def setWinner(self, winner):
        for state, action, player in self.memory:
            self.save_history(state, action, 1 if player == winner else -1)
        self.memory = []

    def clearMemory(self):
        self.memory = []

    def addAixs(self, data):
        return data.reshape(data.shape + (1, ))

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
            tf.keras.layers.Dense(size * size, activation='linear')
        ])
        model.compile(
            loss=tf.losses.mse,
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
        )
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

class ThreadGobang(Thread.ThreadBase):
    saveType = {
        'episode': (str, int),
        'winRate': (json.dumps, json.loads),
    }
    bakFrequence = 2000
    def run(self):
        env = envoriment.Gobang()
        self.env = env
        agent = self.agentType(env)
        config = self.load()
        if config is not None:
            agent.model = config['model']
            episode = config['episode']
            self.winRate = config['winRate']
        else:
            episode = 0
            self.winRate = []
        print(agent.model.summary())
        agentPre = self.agentType(env)
        agentPre.model = copyModel(agent.model)
        win, lose = 0, 0
        while True:
            state = env.reset()
            self.render(env, 0.5)
            step = 0
            s1 = time.time()
            while True:
                action = agent.choose_action(state, 0.9 if self.mode == 0 else 1.0)
                next_state, player, winner = env.step(action)
                agent.save_exp(state, action, player)
                step += 1
                state = next_state
                self.render(env, 0.5)
                if winner is not None:
                    break
            if winner != 0:
                episode += 1
                if winner == 1:
                    win += 1
                elif winner == 2:
                    lose += 1
                if episode % 100 == 0:
                    self.winRate.append(win)
                    win, lose = 0, 0
                agent.setWinner(winner)
                if episode % 5 == 0:
                    s2 = time.time()
                    agent.learn()
                    print('learn spend {} seconds'.format(time.time() - s2))
                    s3 = time.time()
                    result = self.battle(agent, agentPre)
                    if result == 0:
                        agent.model = copyModel(agentPre.model)
                    else:
                        agentPre.model = copyModel(agent.model)
                    print('test train result: {}, spend {} seconds'.format(result, time.time() - s3))
                if episode % 50 == 0:
                    self.save(agent.model, episode=episode, winRate=self.winRate)
            else:
                agent.clearMemory()
            print('episode {}, winer {}, step {}, spend {} seconds'.format(episode, winner, step, time.time() - s1))

    def battle(self, agent, agentPre):
        env = self.env
        state = env.reset()
        agentIdx = random.randint(1, 2)
        agents = [None, None, None]
        agents[agentIdx] = agent
        agents[3 - agentIdx] = agentPre
        player = 1
        while True:
            action = agents[player].choose_action(state, 1.0)
            player = 3 - player
            state, p, winner = env.step(action)
            if winner is not None:
                break
        if winner == 0 or winner == agentIdx:
            return 1
        else:
            return 0

    def load(self):
        config = None
        if self.savePath and os.path.exists(self.savePath):
            config = self.readConfig(os.path.join(self.savePath, 'config.txt'))
            modelPath = os.path.join(self.savePath, 'model_{}-{}.h5'.format(
                (config['episode'] // self.bakFrequence) * self.bakFrequence,
                ((config['episode'] // self.bakFrequence) + 1) * self.bakFrequence,
            ))
            model = tf.keras.models.load_model(modelPath)
            config['model'] = model
        return config

    def save(self, model, episode, **kwargs):
        if self.savePath:
            if not os.path.exists(self.savePath):
                os.mkdir(self.savePath)
            modelPath = os.path.join(self.savePath, 'model_{}-{}.h5'.format(
                (episode // self.bakFrequence) * self.bakFrequence,
                ((episode // self.bakFrequence) + 1) * self.bakFrequence,
            ))
            model.save(modelPath)
            configPath = os.path.join(self.savePath, 'config.txt')
            kwargs['episode'] = episode
            with open(configPath, 'w') as f:
                for k, v in kwargs.items():
                    f.write('{}={}\n'.format(k, self.saveType[k][0](v)))

    def readConfig(self, filePath):
        config = {}
        with open(filePath) as f:
            line = f.readline().strip()
            while line:
                idx = line.find('=')
                if idx != -1:
                    name = line[:idx]
                    data = line[idx + 1:]
                    config[name] = self.saveType[name][1](data)
                line = f.readline()
        return config

    def showWinRate(self):
        Image(self.winRate)

if __name__ == '__main__':
    thread = ThreadGobang(Agent, showProcess=True, mode=1, savePath=getDataFilePath('GoBang1'))
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


