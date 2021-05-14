import copy
import json

import numpy as np

from lib import *
import matplotlib.animation as animation
import time
import random
import Thread
import envoriment
import math

class modelAgent(tf.keras.Model):
    
    def __init__(self, size):
        super(modelAgent, self).__init__()
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(size * size, activation='softmax')

    def call(self, inputs, training=False, mask=None):
        out = self.flatten1(inputs)
        out = self.dense1(out, training=training)
        out = self.dropout1(out, training=training)
        out = self.dense2(out, training=training)
        out = self.dropout2(out, training=training)
        out = self.dense3(out, training=training)
        return out

'''
model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(size, size)),
            tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(size * size, activation='softmax')
        ])
'''
class Agent(object):
    def __init__(self, env):
        self.env = env
        self.model = self._createModel(self.env.SIZE)
        self.exp = []
        self.temp_exp = []
        self.memorySize = 1000
        self.memoryIter = 0
        self.memory = []

    def choose_action(self, state, choose_max=False):
        actions = self.env.validActions(state)
        prop = self.model.predict(np.array([self.addAixs(state)]))[0]
        if choose_max:
            maxi, maxv = actions[0], prop[actions[0]]
            for i, v in enumerate(prop):
                if i in actions and v > maxv:
                    maxi, maxv = i, v
            return maxi
        else:
            prop = [0 if i not in actions else v for i, v in enumerate(prop)]
            return np.random.choice(self.env.ACTION_SIZE, p=normallize(prop))

    def saveExp(self, state, action, player):
        self.temp_exp.append((state, action, player))

    def clearExp(self):
        self.temp_exp = []

    def rememberExp(self):
        for e in self.exp:
            self.saveMemory(*e)
        self.exp = []

    def forgetExp(self):
        self.exp = []

    def clearMemory(self):
        self.memory = []
        self.memoryIter = 0

    def forgetMemory(self, size=1):
        if len(self.memory) > 0:
            n = np.random.choice(len(self.memory), size=size)
            mem = []
            for i, m in enumerate(self.memory):
                if i not in n:
                    mem.append(m)
            self.memory = mem

    def saveMemory(self, state, action, value):
        if len(self.memory) < self.memorySize:
            self.memory.append((state, action, value))
        else:
            self.memory[self.memoryIter] = (state, action, value)
            self.memoryIter = (self.memoryIter + 1) % self.memorySize

    def sample(self, batch):
        res = []
        if len(self.exp) + len(self.memory) < batch:
            batch = len(self.exp) + len(self.memory)
            res.extend(sample(self.exp, len(self.exp)))
            res.extend(sample(self.memory, len(self.memory)))
        elif len(self.exp) < batch // 2:
            res.extend(sample(self.exp, len(self.exp)))
            res.extend(sample(self.memory, batch - len(self.exp)))
        elif len(self.memory) < batch // 2:
            res.extend(sample(self.memory, len(self.memory)))
            res.extend(sample(self.exp, batch - len(self.memory)))
        else:
            res.extend(sample(self.memory, batch // 2))
            res.extend(sample(self.exp, batch // 2))
        random.shuffle(res)
        return [[res[x][i] for x in range(batch)] for i in range(3)]

    def learn(self, batch=128):
        S, A, V = self.sample(batch)
        states, actions, values = [], [], []
        for s, a, v in zip(S, A, V):
            s, a = self.env.extendState(s, a)
            states.extend(s)
            actions.extend(a)
            values.extend([v] * len(s))
        states, actions, values = np.array(states), np.array(actions), np.array(values)
        states = self.addAixs(states)
        predict = self.model.predict(states)
        for i, p, a, v in zip(range(len(predict)), predict, actions, values):
            p[a] = v
            predict[i] = normallize(p)
        self.model.fit(states, predict, epochs=10, verbose=0)

    def setWinner(self, winner):
        for state, action, player in self.temp_exp:
            self.exp.append((state, action, 1 if player == winner else 0))
        self.clearExp()

    def addAixs(self, data):
        return data
        # return data.reshape(data.shape + (1, ))

    def _createModel(self, size, learning_rate=0.001):
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=(size, size, 1)),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(1024, activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.3),
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.3),
        #     tf.keras.layers.Dense(size * size, activation='linear')
        # ])
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Flatten(input_shape=(size, size)),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     # tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     # tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(size * size, activation='softmax')
        # ])
        inputs = tf.keras.Input(shape=(size, size))
        outputs = tf.keras.layers.Flatten()(inputs)
        outputs = tf.keras.layers.Dense(128, activation='relu')(outputs)
        outputs = tf.keras.layers.Dropout(0.2)(outputs)
        outputs = tf.keras.layers.Dense(128, activation='relu')(outputs)
        outputs = tf.keras.layers.Dropout(0.2)(outputs)
        outputs = tf.keras.layers.Dense(size * size, activation='softmax')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
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
    bakFrequence = 500
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
            path = []
            while True:
                action = agent.choose_action(state, bool(self.mode))
                next_state, player, winner = env.step(action)
                agent.saveExp(state, action, player)
                step += 1
                state = next_state
                path.append(action)
                self.render(env, 0.5)
                if winner is not None:
                    break
            print('path {}\nepisode {}, winner {}, step {}, spend {} seconds'.format(path, episode, winner, step, time.time() - s1))
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
                if episode % 1 == 0:
                    s2 = time.time()
                    agent.learn(32)
                    print('learn spend {} seconds'.format(time.time() - s2))
                if episode % 1 == 0:
                    s3 = time.time()
                    improve = self.isImprove(agent, agentPre)
                    if not improve:
                        agent.model = self.save(agentPre.model, episode=episode, winRate=self.winRate)
                        agent.forgetExp()
                        agent.forgetMemory(1)
                    else:
                        agentPre.model = self.save(agent.model, episode=episode, winRate=self.winRate)
                        agent.rememberExp()
                    print('train result: {}, spend {} seconds'.format(int(improve), time.time() - s3))

            else:
                agent.clearExp()

    def isImprove(self, agent, agentPre):
        env = self.env
        win, lose = 0, 0
        for agentIdx in range(1, 3):
            for i in range(10):
                state = env.reset()
                agents = [None, None, None]
                agents[agentIdx] = agent
                agents[3 - agentIdx] = agentPre
                player = 1
                while True:
                    action = agents[player].choose_action(state, True)
                    player = 3 - player
                    state, p, winner = env.step(action)
                    if winner is not None:
                        break
                if winner != 0:
                    if winner == agentIdx:
                        win += 1
                    else:
                        lose += 1
        return win / (win + lose) > 0.5

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
            return tf.keras.models.load_model(modelPath)
        else:
            return copyModel(model)

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
    thread = ThreadGobang(Agent, showProcess=False, mode=0, savePath=getDataFilePath('GoBang'))
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


