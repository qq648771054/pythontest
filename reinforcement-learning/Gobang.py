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
        self.Es = {}        # stores end state values

    def chooseAction(self, state, selectMax=False, episode=20):
        s = self.env.board2Str(state)
        for i in range(episode):
            self.search(state)
        actions = self.env.validActions(state)
        counts = [self.Nsa[s].get(i, 0) if i in actions else 0 for i in range(self.env.ACTION_SIZE)]
        counts = normallize(counts)
        if selectMax:
            maxA = actions[0]
            for a in actions:
                if self.Nsa[s].get(maxA, 0) < self.Nsa[s].get(a, 0):
                    maxA = a
            return maxA, counts
        else:
            return np.random.choice(self.env.ACTION_SIZE, p=counts), counts

    def search(self, state):
        s = self.env.board2Str(state)
        if s not in self.Es:
            winner = self.env.getWiner(state)
            if winner is None:
                self.Es[s] = 0
            elif winner == 1:
                self.Es[s] = 1
            else:
                self.Es[s] = 0.5
        if self.Es[s] != 0:
            return self.Es[s]
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
                u = self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-6)
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
    def __init__(self, env, memroySize):
        self.env = env
        self.model = self._createModel(self.env.SIZE)
        self.temp_exp = []
        self.exp = []
        self.memory = []
        self.memoryWeight = []
        self.memorySize = memroySize
        self.memoryIter = 0
        self.mcts = MCTS(self.env, self)

    def saveExp(self, state, prop, player):
        self.temp_exp.append((state, prop, player))

    def saveMemory(self, scale):
        for m, resultState in self.exp:
            self.appendMemory(m, scale[resultState])
        self.exp = []

    def appendMemory(self, m, weight):
        if len(self.memory) < self.memorySize:
            self.memory.append(m)
            self.memoryWeight.append(weight)
        else:
            self.memory[self.memoryIter] = m
            self.memoryWeight[self.memoryIter] = weight
            self.memoryIter = (self.memoryIter + 1) % self.memorySize

    def discountMemoryWeight(self, discount=0.8):
        for i, w in enumerate(self.memoryWeight):
            self.memoryWeight[i] = w * discount

    def sampleMemory(self, batch):
        if len(self.memory) < batch:
            idx = np.random.choice(len(self.memory), size=int(len(self.memory) * 0.8), p=normallize(self.memoryWeight))
        else:
            idx = np.random.choice(len(self.memory), size=batch, p=normallize(self.memoryWeight))
        return [[self.memory[i][j] for i in idx] for j in range(3)]

    def setWinner(self, winner, resultState):
        for state, prop, player in self.temp_exp:
            if winner == 0:
                self.exp.append(((state, prop, 0.5), resultState))
            else:
                self.exp.append(((state, prop, 1.0 if winner == player else 0.0), resultState))
        self.temp_exp = []

    def chooseAction(self, state, mode=0, mcts=None):
        if mode == 2:
            actions = self.env.validActions(state)
            prop = self.model.predict(np.array([state]))[0][0]
            maxa = actions[0]
            for a in actions:
                if maxa < prop[a]:
                    maxa = a
            return maxa, [0.0 if i != maxa else 1.0 for i in range(self.env.ACTION_SIZE)]
        else:
            if mcts is None:
                mcts = self.mcts
            if mode == 0:
                return mcts.chooseAction(state, False)
            elif mode == 1:
                return mcts.chooseAction(state, True)

    def learn(self, batch):
        S, P, V = self.sampleMemory(batch)
        trainSize = (len(self.env.sameShapes) * len(S))
        states, props, values = ([0] * trainSize), ([0] * trainSize), ([0] * trainSize)
        idx = list(range(trainSize))
        random.shuffle(idx)
        i = 0
        for s, p, v in zip(S, P, V):
            s, p = self.env.extendState(s, p)
            for si, pi in zip(s, p):
                states[idx[i]], props[idx[i]], values[idx[i]] = si, pi, v
                i += 1
        states, props, values = np.array(states), np.array(props), np.array(values)
        states = self.addAixs(states)
        self.model.fit(states, [props, values], epochs=10, verbose=1)
        self.mcts = MCTS(self.env, self)

    def addAixs(self, data):
        return data.reshape(data.shape + (1, ))

    def _createModel(self, size, learning_rate=0.001):
        model = self.modelDic(size)
        model.compile(
            loss=tf.losses.mse,
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
        )
        return model

    def modelDic(self, size):
        if size == 15:
            inputs = tf.keras.Input(shape=(size, size, 1))
            outputs = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(inputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dense(1024, activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            output1 = tf.keras.layers.Dense(size * size, activation='softmax')(outputs)
            output2 = tf.keras.layers.Dense(1, activation='linear')(outputs)
            return tf.keras.Model(inputs=inputs, outputs=[output1, output2])
        elif size == 3:
            inputs = tf.keras.Input(shape=(size, size, 1))
            outputs = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dense(32, activation='relu')(outputs)
            outputs = tf.keras.layers.Dropout(0.2)(outputs)
            outputs = tf.keras.layers.Dense(32, activation='relu')(outputs)
            outputs = tf.keras.layers.Dropout(0.2)(outputs)
            output1 = tf.keras.layers.Dense(size * size, activation='softmax')(outputs)
            output2 = tf.keras.layers.Dense(1, activation='linear')(outputs)
            return tf.keras.Model(inputs=inputs, outputs=[output1, output2])
        elif size == 5:
            inputs = tf.keras.Input(shape=(size, size, 1))
            outputs = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dense(128, activation='relu')(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            outputs = tf.keras.layers.Dense(128, activation='relu')(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            output1 = tf.keras.layers.Dense(size * size, activation='softmax')(outputs)
            output2 = tf.keras.layers.Dense(1, activation='linear')(outputs)
            return tf.keras.Model(inputs=inputs, outputs=[output1, output2])
        elif size == 10:
            inputs = tf.keras.Input(shape=(size, size, 1))
            outputs = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dense(256, activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            output1 = tf.keras.layers.Dense(size * size, activation='softmax')(outputs)
            output2 = tf.keras.layers.Dense(1, activation='linear')(outputs)
            return tf.keras.Model(inputs=inputs, outputs=[output1, output2])

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
'''
showProcess=False, player=0, mode=0,
                          boardSize=5, boardWin=4,
                          savePath=getDataFilePath('GoBang_5_4_1'), logName='log.txt'
'''
class Gobang(object):
    saveType = {
        'episode': (str, int),
        'winRate': (json.dumps, json.loads),
    }
    bakFrequence = 500
    def __init__(self, agentType,
                 showProcess=True, player=0, mode=0,
                 boardSize=15, boardWin=5,
                 savePath='', logName='log.txt'):
        self.agentType = agentType
        self.showProcess = showProcess
        self.player = player
        self.mode = mode
        self.boardSize = boardSize
        self.boardWin = boardWin
        self.savePath = savePath
        self.logName = logName
        self.env = None
        self.lastClick = None
        if self.player != 0:
            self.showProcess = True

    def run(self):
        trainPeriod = 50
        memorySize = trainPeriod * 20 * 10
        batchSize = memorySize // 3
        env = envoriment.Gobang(onGridClick=self.onGridClick)
        env.reshape(self.boardSize, self.boardWin)
        self.env = env
        agent = self.agentType(env, memorySize)
        config = self.load()
        if config is not None:
            agent.model = config['model']
            episode = config['episode']
            self.winRate = config['winRate']
        else:
            episode = 0
            self.winRate = []
        agent.model.summary()
        agentPre = self.agentType(env, memorySize)
        agentPre.model = copyModel(agent.model)
        if self.savePath and hasattr(self, 'logName'):
            self.logFile = open(os.path.join(self.savePath, self.logName), 'a')
            self.logs = []
        win, lose, draw = 0, 0, 0
        while True:
            state = env.reset()
            self.render(0.5)
            step = 0
            s1 = time.time()
            path = []
            currentPlayer = 1
            while True:
                if currentPlayer == self.player:
                    action = self.waitPlayerClick()
                    next_state, player, winner = env.step(action)
                else:
                    action, prop = agent.chooseAction(state, self.mode)
                    next_state, player, winner = env.step(action)
                    agent.saveExp(state, prop, player)
                step += 1
                state = next_state
                currentPlayer = 3 - currentPlayer
                path.append(action)
                self.render(0.5)
                if winner is not None:
                    break
            episode += 1
            self.log('path {}'.format(path))
            self.log('episode {}, winner {}, step {}, spend {} seconds'.format(episode, winner, step, time.time() - s1))
            if winner == 1:
                win += 1
                agent.setWinner(winner, 0)
            elif winner == 2:
                lose += 1
                agent.setWinner(winner, 1)
            else:
                draw += 1
                agent.setWinner(winner, 2)
            if episode % trainPeriod == 0:
                self.winRate.append([episode, win, lose, draw])
                sampleTypes = 0.0
                sampleTypes += 1.0 if win > 0 else 0.0
                sampleTypes += 1.0 if lose > 0 else 0.0
                sampleTypes += 1.0 if draw > 0 else 0.0
                std = trainPeriod / sampleTypes
                winScale = std / win if win > 0 else 1.0
                loseScale = std / lose if lose > 0 else 1.0
                drawScale = std / draw if draw > 0 else 1.0
                agent.saveMemory(scale=[winScale, loseScale, drawScale])
                for i in range(4):
                    s2 = time.time()
                    agent.learn(batchSize)
                    self.log('learn spend {} seconds'.format(time.time() - s2))
                    s3 = time.time()
                    winRate = self.comapreWithPre(agent, agentPre)
                    if winRate < 0.6:
                        agent.model = self.save(agentPre.model, episode=episode, winRate=self.winRate)
                        self.log('train result: {}, spend {} seconds'.format(0, time.time() - s3))
                    else:
                        agentPre.model = self.save(agent.model, episode=episode, winRate=self.winRate)
                        self.log('train result: {}, spend {} seconds'.format(1, time.time() - s3))
                        break
                agent.discountMemoryWeight(0.8)
                win, lose, draw = 0, 0, 0

    def onGridClick(self, x, y):
        self.lastClick = y * self.env.SIZE + x

    def waitPlayerClick(self):
        validActions = self.env.validActions()
        while True:
            self.lastClick = None
            self.render()
            time.sleep(0.015)
            if self.lastClick is not None and self.lastClick in validActions:
                return self.lastClick

    def comapreWithPre(self, agent, agentPre, battleCount=20):
        env = self.env
        win, lose = 0, 0
        episode = 0
        mctsAgent = MCTS(env, agent)
        mctsAgentPre = MCTS(env, agentPre)
        agents = [None, None, None]
        mcts = [None, None, None]
        for i in range(battleCount // 2):
            for agentIdx in range(1, 3):
                agents[agentIdx] = agent
                agents[3 - agentIdx] = agentPre
                mcts[agentIdx] = mctsAgent
                mcts[3 - agentIdx] = mctsAgentPre
                state = env.reset()
                player = 1
                step = []
                while True:
                    action, prop = agents[player].chooseAction(state, 1, mcts=mcts[player])
                    player = 3 - player
                    state, p, winner = env.step(action)
                    step.append(action)
                    if winner is not None:
                        break
                if winner == 0:
                    if agentIdx == 1:
                        lose += 0.1
                    else:
                        win += 0.1
                elif winner == agentIdx:
                    win += 1
                else:
                    lose += 1
                episode += 1
                self.log('self battle {}: path {}, expect {}, winner {}'.format(episode, step, agentIdx, winner))
        self.log('self battle win rate: {}'.format(win / (win + lose) if win + lose > 0 else 0))
        return win / (win + lose) if win + lose > 0 else 0

    def load(self):
        config = None
        if self.savePath:
            if os.path.exists(os.path.join(self.savePath, 'config.txt')):
                config = self.readConfig(os.path.join(self.savePath, 'config.txt'))
                modelPath = os.path.join(self.savePath, 'model_{}-{}.h5'.format(
                    (config['episode'] // self.bakFrequence) * self.bakFrequence,
                    ((config['episode'] // self.bakFrequence) + 1) * self.bakFrequence,
                ))
                model = tf.keras.models.load_model(modelPath)
                config['model'] = model
            elif not os.path.exists(self.savePath):
                os.mkdir(self.savePath)
        return config

    def save(self, model, episode, **kwargs):
        if hasattr(self, 'logFile') and len(self.logs) > 0:
            for log in self.logs:
                self.logFile.write(log)
            self.logFile.flush()
            self.logs = []
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

    def log(self, s):
        print(s)
        if hasattr(self, 'logFile'):
            self.logs.append(s)
            self.logs.append('\n')

    def render(self, sleepTime=None):
        if self.showProcess:
            self.env.render()
            if sleepTime:
                time.sleep(sleepTime)


if __name__ == '__main__':
    game = Gobang(Agent, showProcess=False, player=0, mode=0,
                          boardSize=5, boardWin=4,
                          savePath=getDataFilePath('GoBang_5_4_1'), logName='log.txt')

    # game = Gobang(Agent, showProcess=False, player=2, mode=1,
    #                       boardSize=3, boardWin=3,
    #                       savePath=getDataFilePath('GoBang_3_3_1'), logName='log.txt')
    game.run()


