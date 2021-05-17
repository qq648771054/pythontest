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

class MCTS(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

    def chooseAction(self, state, selectMax=False, episode=20):
        s = self.env.board2Str(state)
        for i in range(episode):
            self.search(state)
        actions = self.env.validActions(state)
        if selectMax:
            maxA = actions[0]
            for a in actions:
                if self.Nsa[s].get(maxA, 0) < self.Nsa[s].get(a, 0):
                    maxA = a
            return maxA
        else:
            counts = [self.Nsa[s].get(i, 0) if i in actions else 0 for i in range(self.env.ACTION_SIZE)]
            return np.random.choice(self.env.ACTION_SIZE, p=normallize(counts))

    def search(self, state):
        s = self.env.board2Str(state)
        winner = self.env.getWiner(state)
        if winner is not None:
            if winner == 1:
                return 1
            else:
                return 0.5
        actions = self.env.validActions(state)
        if s not in self.Ps:
            self.Ps[s], v = self.agent.model.predict(self.agent.addAixs(np.array([state])))
            self.Ps[s], v = self.Ps[s][0], clamp(v[0][0], 0, 1)
            self.Ps[s] = [self.Ps[s][i] if i in actions else 0 for i in range(self.env.ACTION_SIZE)]
            self.Ps[s] = normallize(self.Ps[s])
            return v
        if s not in self.Ns:
            self.Ns[s] = 0
            self.Qsa[s] = {}
            self.Nsa[s] = {}
        bu, ba = -float('inf'), -1
        for a in actions:
            if a in self.Qsa[s]:
                u = self.Qsa[s][a] + self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[s][a])
            else:
                u = self.Ps[s][a] * math.sqrt(self.Ns[s])
            if u > bu:
                bu = u
                ba = a
        next_state = self.env.getNextState(state, ba)
        v = self.search(next_state)
        if ba in self.Qsa[s]:
            self.Qsa[s][ba] = (self.Nsa[s][ba] * self.Qsa[s][ba] + v) / (self.Nsa[s][ba] + 1)
            self.Nsa[s][ba] += 1
        else:
            self.Qsa[s][ba] = v
            self.Nsa[s][ba] = 1
        self.Ns[s] += 1
        return 1 - v

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.model = self._createModel(self.env.SIZE)
        self.exp = []
        self.memory = []
        self.mcts = MCTS(self.env, self)

    def saveExp(self, state, action, player):
        self.exp.append((state, action, player))

    def clearExp(self):
        self.exp = []

    def clearMemory(self):
        self.memory = []
        self.mcts = MCTS(self.env, self)

    def setWinner(self, winner):
        for state, action, player in self.exp:
            if winner == 0:
                self.memory.append((state, action, 0.5))
            else:
                self.memory.append((state, action, 1 if player == winner else 0))
        self.clearExp()

    def chooseAction(self, state, selectMax=False, mcts=None):
        if mcts is None:
            return self.mcts.chooseAction(state, selectMax)
        else:
            return mcts.chooseAction(state, selectMax)

    def learn(self):
        S, A, V = [[self.memory[i][j] for i in range(len(self.memory))] for j in range(3)]
        states, actions, values = [], [], []
        for s, a, v in zip(S, A, V):
            s, a = self.env.extendState(s, a)
            states.extend(s)
            actions.extend(a)
            values.extend([v] * len(s))
        states, actions, values = np.array(states), np.array(actions), np.array(values)
        states = self.addAixs(states)
        prop, value = self.model.predict(states)
        for i, p, a, v in zip(range(len(prop)), prop, actions, values):
            p[a] = v
            prop[i] = normallize(p)
            value[i] = [v]
        self.model.fit(states, [prop, value], epochs=10, verbose=0)
        self.clearMemory()

    def addAixs(self, data):
        return data.reshape(data.shape + (1, ))

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
        inputs = tf.keras.Input(shape=(size, size, 1))
        outputs = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(inputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        # outputs = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(outputs)
        # outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Flatten()(outputs)
        outputs = tf.keras.layers.Dense(1024, activation='relu')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Dropout(0.3)(outputs)
        outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Dropout(0.3)(outputs)
        output1 = tf.keras.layers.Dense(size * size, activation='softmax')(outputs)
        output2 = tf.keras.layers.Dense(1, activation='linear')(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])
        # inputs = tf.keras.Input(shape=(size, size))
        # outputs = tf.keras.layers.Flatten()(inputs)
        # outputs = tf.keras.layers.Dense(32, activation='relu')(outputs)
        # outputs = tf.keras.layers.Dropout(0.2)(outputs)
        # outputs = tf.keras.layers.Dense(32, activation='relu')(outputs)
        # outputs = tf.keras.layers.Dropout(0.2)(outputs)
        # output1 = tf.keras.layers.Dense(size * size, activation='softmax')(outputs)
        # output2 = tf.keras.layers.Dense(1, activation='linear')(outputs)
        # model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])
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
                action = agent.chooseAction(state, bool(self.mode))
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
                if winner == 1:
                    win += 1
                elif winner == 2:
                    lose += 1
                if win + lose == 10:
                    self.winRate.append(win)
                    win, lose = 0, 0
            episode += 1
            agent.setWinner(winner)
            if episode % 20 == 0:
                s2 = time.time()
                agent.learn()
                print('learn spend {} seconds'.format(time.time() - s2))
                # self.save(agent.model, episode=episode, winRate=self.winRate)
                s3 = time.time()
                improve = self.isImprove(agent, agentPre)
                if not improve:
                    agent.model = self.save(agentPre.model, episode=episode, winRate=self.winRate)
                else:
                    agentPre.model = self.save(agent.model, episode=episode, winRate=self.winRate)
                print('train result: {}, spend {} seconds'.format(int(improve), time.time() - s3))

    def isImprove(self, agent, agentPre, battleCount=10):
        env = self.env
        win, lose = 0, 0
        episode = 0
        mctsAgent = MCTS(env, agent)
        mctsAgentPre = MCTS(env, agentPre)
        for agentIdx in range(1, 3):
            agents = [None, None, None]
            mcts = [None, None, None]
            agents[agentIdx] = agent
            agents[3 - agentIdx] = agentPre
            mcts[agentIdx] = mctsAgent
            mcts[3 - agentIdx] = mctsAgentPre
            for i in range(battleCount // 2):
                state = env.reset()
                player = 1
                step = []
                while True:
                    action = agents[player].chooseAction(state, True, mcts=mcts[player])
                    player = 3 - player
                    state, p, winner = env.step(action)
                    step.append(action)
                    if winner is not None:
                        break
                if winner != 0:
                    if winner == agentIdx:
                        win += 1
                    else:
                        lose += 1
                episode += 1
                print('self battle {}: path {}, winer{}'.format(episode, step, winner))
        print('self battle win rate: {}:'.format(win / (win + lose)))
        return win / (win + lose) >= 0.55

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
    # thread = ThreadGobang(Agent, showProcess=False, mode=0)
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


