import math

from lib import *
import matplotlib.animation as animation
import time
import datetime
import random
import Game
import json

MODE_GREDDY = 0
MODE_RANDOM1 = 1
MODE_RANDOM2 = 2

class Agent(object):
    def __init__(self, env, memroySize):
        self.env = env
        self.model = self._createModel(self.env.SIZE)
        self.temp_exp = []
        self.exp = []
        if isinstance(memroySize, int):
            memroySize = [memroySize]
        self.memorySize = memroySize
        self.memoryAxis = len(self.memorySize)
        self.memory = [[] for i in range(self.memoryAxis)]
        self.memoryIter = [0] * self.memoryAxis
        self.memoryWeight = [[] for i in range(len(self.memorySize))]

    def clearAll(self):
        self.memory = [[] for i in range(len(self.memorySize))]
        self.memoryIter = [0] * self.memoryAxis
        self.memoryWeight = [[] for i in range(len(self.memorySize))]
        self.exp = []

    def saveExp(self, state, action, player):
        self.exp.append((state, action, player))

    def appendMemory(self, axis, m, weight):
        if len(self.memory[axis]) < self.memorySize[axis]:
            self.memory[axis].append(m)
            self.memoryWeight[axis].append(weight)
        else:
            self.memory[axis][self.memoryIter[axis]] = m
            self.memoryWeight[axis][self.memoryIter[axis]] = weight
            self.memoryIter[axis] = (self.memoryIter[axis] + 1) % self.memorySize[axis]

    def discountMemoryWeight(self, discount=0.8):
        for weight in self.memoryWeight:
            for i, w in enumerate(weight):
                weight[i] = w * discount

    def sampleMemory(self, batch):
        def sample(m, mw, b):
            if len(m) > 0:
                idx = np.random.choice(len(m), size=b, p=normallize(mw))
                return [[m[i][j] for i in idx] for j in range(3)]
            else:
                return [], [], []
        sa = [[], [], []]
        for i in range(self.memoryAxis):
            s = sample(self.memory[i], self.memoryWeight[i], batch[i])
            sa[0] += s[0]
            sa[1] += s[1]
            sa[2] += s[2]
        return sa[0], sa[1], sa[2]

    def setWinner(self, winner, axis, isFirst):
        for state, action, player in self.exp:
            if winner == 0:
                self.appendMemory(axis, (state, action, 0.4 if isFirst else 0.6), 1.0)
            else:
                self.appendMemory(axis, (state, action, 1.0 if winner == player else 0.0), 1.0)
        self.exp = []

    def chooseAction(self, state, mode=MODE_GREDDY):
        values = clamp(self.model.predict(np.array([state]))[0], 0.0001, 0.9999)
        actions = self.env.validActions(state, player=2)
        if mode == MODE_GREDDY:
            maxA = actions[0]
            for a in actions:
                if values[a] > values[maxA]:
                    maxA = a
            return maxA
        elif mode == MODE_RANDOM1:
            values = normallize([values[i] ** 3 if i in actions else 0 for i in range(self.env.ACTION_SIZE)])
            return np.random.choice(self.env.ACTION_SIZE, p=values)
        elif mode == MODE_RANDOM2:
            if random.random() >= 0.95:
                return random.choice(actions)
            else:
                values = normallize([values[i] if i in actions else 0 for i in range(self.env.ACTION_SIZE)])
                return np.random.choice(self.env.ACTION_SIZE, p=values)

    def learn(self, batch, epochs=10):
        S, A, V = self.sampleMemory(batch)
        P = self.model.predict(np.array(S))
        trainSize = (len(self.env.sameShapes) * len(S))
        states, props = ([0] * trainSize), ([0] * trainSize)
        idx = list(range(trainSize))
        random.shuffle(idx)
        i = 0
        for s, a, p, v in zip(S, A, P, V):
            p = clamp(p, 0, 1)
            p[a] = v
            s, p = self.env.extendState(s, p)
            for si, pi in zip(s, p):
                states[idx[i]], props[idx[i]] = si, pi
                i += 1
        states, props = np.array(states), np.array(props)
        states = self.addAxis(states)
        self.model.fit(states, props, epochs=epochs, verbose=1)

    def addAxis(self, data):
        return data.reshape(data.shape + (1, ))

    def _createModel(self, size, learning_rate=0.001):
        model = self.modelDict(size)
        model.compile(
            loss=tf.losses.mse,
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
        )
        return model

    def modelDict(self, size):
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
            output1 = tf.keras.layers.Dense(size * size, activation='linear')(outputs)
            return tf.keras.Model(inputs=inputs, outputs=output1)
        elif size == 3:
            inputs = tf.keras.Input(shape=(size, size, 1))
            outputs = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dense(128, activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            outputs = tf.keras.layers.Dense(128, activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            output1 = tf.keras.layers.Dense(size * size, activation='linear')(outputs)
            return tf.keras.Model(inputs=inputs, outputs=output1)
        elif size == 5:
            inputs = tf.keras.Input(shape=(size, size, 1))
            outputs = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            outputs = tf.keras.layers.Dense(256, activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            output1 = tf.keras.layers.Dense(size * size, activation='linear')(outputs)
            return tf.keras.Model(inputs=inputs, outputs=output1)
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
            output1 = tf.keras.layers.Dense(size * size, activation='linear')(outputs)
            return tf.keras.Model(inputs=inputs, outputs=output1)

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
                 boardSize=15, boardWin=5,
                 savePath=''):
        self.agentType = agentType
        self.boardSize = boardSize
        self.boardWin = boardWin
        self.savePath = savePath
        self.env = Game.Gobang(onGridClick=self.onGridClick)
        self.env.reshape(self.boardSize, self.boardWin)
        self.lastClick = None

    def train(self, showProcess=False):
        trainPeriod = 20
        memorySize = trainPeriod * (self.env.ACTION_SIZE // 2) * 5
        batchSize = memorySize // 2
        agent = self.agentType(self.env, [memorySize // 4, memorySize // 4, memorySize // 4, memorySize // 4])
        config = self.load()
        if config is not None:
            agent.model = config['model']
            episode = config['episode']
            self.winRate = config['winRate']
        else:
            episode = 0
            self.winRate = []
        agent.model.summary()
        agentPre = self.agentType(self.env, 0)
        agentPre.model = copyModel(agent.model)
        self.logs = []
        win, lose, draw1, draw2 = 0, 0, 0, 0
        self.log('train start')
        agentPlayer = 2
        maxRate = 0
        bakModel = copyModel(agent.model)
        while True:
            state = self.env.reset()
            showProcess and self.render(0.5)
            step = 0
            s1 = time.time()
            path = []
            currentPlayer = 1
            agentPlayer = 3 - agentPlayer
            mode = random.choice([MODE_RANDOM1, MODE_RANDOM1, MODE_RANDOM2])
            while True:
                if currentPlayer == agentPlayer:
                    action = agent.chooseAction(state, mode)
                    next_state, player, winner = self.env.step(action)
                    agent.saveExp(state, action, player)
                else:
                    action = agentPre.chooseAction(state, MODE_GREDDY)
                    next_state, player, winner = self.env.step(action)
                step += 1
                state = next_state
                currentPlayer = 3 - currentPlayer
                path.append(action)
                showProcess and self.render(0.5)
                if winner is not None:
                    break
            episode += 1
            self.log('path {}'.format(path))
            self.log('episode {}, winner {}, expect {}, result {}, step {}, spend {} seconds'.format(
                episode, winner, agentPlayer, int(winner == agentPlayer), step, time.time() - s1))
            if winner == 0:
                if agentPlayer == 1:
                    draw1 += 1
                    agent.setWinner(winner, 2, True)
                else:
                    draw2 += 1
                    agent.setWinner(winner, 3, False)
            elif winner == agentPlayer:
                win += 1
                agent.setWinner(winner, 0, agentPlayer == 1)
            else:
                lose += 1
                agent.setWinner(winner, 1, agentPlayer == 1)
            if episode % trainPeriod == 0:
                rate = (win + draw2 * 0.1) / ((lose + draw1 * 0.1) + (win + draw2 * 0.1))
                self.log('win result: win {}, lose {}, draw1 {}, draw2 {}'.format(win, lose, draw1, draw2))
                if mode == MODE_RANDOM1 and rate >= 0.8:
                    self.log('train 1, win rate {}, mode {}'.format(rate, mode))
                    agentPre.model = self.save(agent.model, episode=episode, winRate=self.winRate)
                    agent.clearAll()
                    self.saveLog(temp=False)
                    maxRate = 0
                else:
                    self.log('train 0, win rate {}, mode {}'.format(rate, mode))
                    if mode == MODE_RANDOM1:
                        if rate < maxRate * 0.75:
                            agent.model = copyModel(bakModel)
                            self.log('roll back model')
                        elif rate > maxRate:
                            maxRate = rate
                            bakModel = copyModel(agent.model)
                    s2 = time.time()
                    agent.learn([batchSize // 3, batchSize // 3, batchSize // 6, batchSize // 6], 10)
                    self.log('learn {} samples, epochs {}, taskes {} seconds'.format(batchSize, 10, time.time() - s2))
                    self.saveLog(temp=True)
                agent.discountMemoryWeight(0.7)
                self.winRate.append([episode, win, lose, draw1, draw2])
                win, lose, draw1, draw2 = 0, 0, 0, 0

    def play(self, player):
        agent = self.agentType(self.env, 0)
        config = self.load()
        if config is not None:
            agent.model = config['model']
        agent.model.summary()
        while True:
            state = self.env.reset()
            self.render(0.5)
            path = []
            currentPlayer = 1
            while True:
                if currentPlayer == player:
                    action = self.waitPlayerClick(player)
                    next_state, p, winner = self.env.step(action)
                else:
                    action = agent.chooseAction(state, MODE_GREDDY)
                    next_state, p, winner = self.env.step(action)
                state = next_state
                currentPlayer = 3 - currentPlayer
                path.append(action)
                self.render(0.5)
                if winner is not None:
                    break
            print('path {}'.format(path))


    def onGridClick(self, x, y):
        self.lastClick = y * self.env.SIZE + x

    def waitPlayerClick(self, player):
        validActions = self.env.validActions(player=player)
        while True:
            self.lastClick = None
            self.render()
            time.sleep(0.015)
            if self.lastClick is not None and self.lastClick in validActions:
                return self.lastClick

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
        print(str(datetime.datetime.now()), ': ', s)
        if self.savePath:
            if not hasattr(self, 'logFile'):
                self.logFile = open(os.path.join(self.savePath, 'log.txt'), 'a')
            self.logs.append(str(datetime.datetime.now()))
            self.logs.append(': ')
            self.logs.append(s)
            self.logs.append('\n')

    def saveLog(self, temp=False):
        if len(self.logs) > 0:
            if hasattr(self, 'logFileTemp'):
                self.logFileTemp.close()
            self.logFileTemp = open(os.path.join(self.savePath, 'temp_log.txt'), 'w')
            if temp:
                self.logFileTemp.write(''.join(self.logs))
                self.logFileTemp.flush()
            else:
                self.logFile.write(''.join(self.logs))
                self.logFile.flush()
                self.logs = []

    def render(self, sleepTime=None):
        self.env.render()
        if sleepTime:
            time.sleep(sleepTime)

    def testAgent(self, a1='', a2=''):
        agent1 = self.agentType(self.env, 0)
        if a1: agent1.model = tf.keras.models.load_model(a1)
        agent2 = self.agentType(self.env, 0)
        if a2: agent2.model = tf.keras.models.load_model(a2)
        agents = [None, None, None]
        episode = 0
        win, lose = 0, 0
        while True:
            for agentIdx in range(1, 3):
                agents[agentIdx] = agent1
                agents[3 - agentIdx] = agent2
                state = self.env.reset()
                player = 1
                step = []
                while True:
                    action = agents[player].chooseAction(state, MODE_GREDDY)
                    player = 3 - player
                    state, p, winner = self.env.step(action)
                    step.append(action)
                    if winner is not None:
                        break
                episode += 1
                if winner == agentIdx:
                    win += 1
                elif winner != 0:
                    lose += 1
                print('self battle {}: path {}, expect {}, winner {}'.format(episode, step, agentIdx, winner))
                print('total: {} win, {} lose'.format(win, lose))


if __name__ == '__main__':
    # game = Gobang(Agent, showProcess=False, player=0, mode=2,
    #                       boardSize=5, boardWin=4,
    #                       savePath=getDataFilePath('GoBang_5_4_3'), logName='log.txt')

    game = Gobang(Agent, boardSize=3, boardWin=3,
                savePath=getDataFilePath('GoBang_3_3_10'))

    # game = Gobang(Agent, boardSize=5, boardWin=4,
    #               savePath=getDataFilePath('GoBang_5_4_1'))

    # game = Gobang(Agent, showProcess=False, player=1, mode=MODE_GREDDY,
    #               boardSize=3, boardWin=3,
    #               savePath=getDataFilePath('GoBang_3_3_1'), logName='log.txt')

    # game = Gobang(Agent, showProcess=False, player=0, mode=0,
    #                       boardSize=3, boardWin=3,
    #                       savePath=getDataFilePath('GoBang_3_3_2'), logName='log.txt')
    game.train()
    # game.play(1)
    # game.testAgent(getDataFilePath('GoBang_5_4_1/model_1000-1500.h5'), getDataFilePath('GoBang_5_4_1/model_0-500.h5'))


