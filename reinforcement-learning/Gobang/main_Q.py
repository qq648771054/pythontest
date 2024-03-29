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
        self.memory = []
        self.memoryWeight = []
        self.memorySize = memroySize
        self.memoryIter = 0

    def saveExp(self, state, action, player):
        self.temp_exp.append((state, action, player))

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
        for state, action, player in reversed(self.temp_exp):
            if winner == 0:
                self.exp.append(((state, action, 0.5), resultState))
            else:
                self.exp.append(((state, action, 1.0 if winner == player else 0.0), resultState))
        self.temp_exp = []

    def chooseAction(self, state, mode=MODE_GREDDY):
        values = clamp(self.model.predict(np.array([state]))[0], 0, 1)
        actions = self.env.validActions(state, player=2)
        if mode == MODE_GREDDY:
            maxA = actions[0]
            for a in actions:
                if values[a] > values[maxA]:
                    maxA = a
            return maxA
        else:
            if mode == MODE_RANDOM2 and random.random() >= 0.9:
                return random.choice(actions)
            else:
                values = normallize([values[i] ** 2 if i in actions else 0 for i in range(self.env.ACTION_SIZE)])
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
        states = self.addAixs(states)
        self.model.fit(states, props, epochs=epochs, verbose=1)

    def addAixs(self, data):
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
            outputs = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dense(64, activation='relu')(outputs)
            outputs = tf.keras.layers.BatchNormalization()(outputs)
            outputs = tf.keras.layers.Dropout(0.3)(outputs)
            outputs = tf.keras.layers.Dense(64, activation='relu')(outputs)
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
        self.env = Game.Gobang(onGridClick=self.onGridClick)
        self.env.reshape(self.boardSize, self.boardWin)
        self.lastClick = None
        if self.player != 0:
            self.showProcess = True

    def run(self):
        trainPeriod = 50
        memorySize = trainPeriod * self.env.ACTION_SIZE * 5
        batchSize = memorySize // 2
        agent = self.agentType(self.env, memorySize)
        config = self.load()
        if config is not None:
            agent.model = config['model']
            episode = config['episode']
            self.winRate = config['winRate']
        else:
            episode = 0
            self.winRate = []
        agent.model.summary()
        agentPre = self.agentType(self.env, memorySize)
        agentPre.model = copyModel(agent.model)
        if self.savePath and hasattr(self, 'logName'):
            self.logFile = open(os.path.join(self.savePath, self.logName), 'a')
            self.logs = []
        win, lose, draw = 0, 0, 0
        self.log('train start')
        while True:
            state = self.env.reset()
            self.render(0.5)
            step = 0
            s1 = time.time()
            path = []
            currentPlayer = 1
            while True:
                if currentPlayer == self.player:
                    action = self.waitPlayerClick(currentPlayer)
                    next_state, player, winner = self.env.step(action)
                else:
                    action = agent.chooseAction(state, self.mode)
                    next_state, player, winner = self.env.step(action)
                    agent.saveExp(state, action, player)
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
                # agent.saveMemory(scale=[math.sqrt(winScale), math.sqrt(loseScale), math.sqrt(drawScale)])
                # agent.saveMemory(scale=[winScale, loseScale, drawScale])
                # scales = [[2, 1, 1], [1, 2, 1], []]
                agent.saveMemory(scale=[math.sqrt(winScale), math.sqrt(loseScale), math.sqrt(drawScale)])
                # agent.saveMemory(scale=[1.0, 1.0, 1.0])
                for i in range(5):
                    s2 = time.time()
                    bs = int(batchSize * (0.5 + random.random()))
                    es = int(5 + 6 * random.random())
                    agent.learn(bs, es)
                    self.log('learn {} memorys, epoches {}, spend {} seconds'.format(bs, es, time.time() - s2))
                    s3 = time.time()
                    winRate = self.comapreWithPre(agent, agentPre)
                    if winRate < 0.55:
                        self.log('train result: {}, spend {} seconds'.format(0, time.time() - s3))
                        agent.model = self.save(agentPre.model, episode=episode, winRate=self.winRate)
                    else:
                        self.log('train result: {}, spend {} seconds'.format(1, time.time() - s3))
                        agentPre.model = self.save(agent.model, episode=episode, winRate=self.winRate)
                        break
                agent.discountMemoryWeight(0.7)
                win, lose, draw = 0, 0, 0

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

    def comapreWithPre(self, agent, agentPre, battleCount=20):
        win, lose = 0, 0
        episode = 0
        agent1 = self.agentType(self.env, 0)
        agent2 = self.agentType(self.env, 0)
        agent1.model = agent.model
        agent2.model = agentPre.model
        agents = [None, None, None]
        mode = MODE_GREDDY
        for i in range(battleCount // 2):
            for agentIdx in range(1, 3):
                agents[agentIdx] = agent1
                agents[3 - agentIdx] = agent2
                state = self.env.reset()
                player = 1
                step = []
                while True:
                    action = agents[player].chooseAction(state, mode)
                    player = 3 - player
                    state, p, winner = self.env.step(action)
                    step.append(action)
                    if winner is not None:
                        break
                if winner == 0:
                    if agentIdx == 1:
                        lose += 0.01
                    else:
                        win += 0.01
                elif winner == agentIdx:
                    win += 1
                else:
                    lose += 1
                episode += 1
                self.log('self battle {} mode {}: path {}, expect {}, winner {}'.format(episode, mode, step, agentIdx, winner))
            mode = MODE_RANDOM1
        self.log('self battle win rate: {}'.format(win / (win + lose)))
        return win / (win + lose)

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
        print(str(datetime.datetime.now()), ': ', s)
        if hasattr(self, 'logFile'):
            self.logs.append(str(datetime.datetime.now()))
            self.logs.append(': ')
            self.logs.append(s)
            self.logs.append('\n')

    def render(self, sleepTime=None):
        if self.showProcess:
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

    game = Gobang(Agent, showProcess=False, player=0, mode=MODE_RANDOM2,
                          boardSize=3, boardWin=3,

                          savePath=getDataFilePath('GoBang_3_3_3'), logName='log.txt')

    # game = Gobang(Agent, showProcess=False, player=1, mode=MODE_GREDDY,
    #               boardSize=3, boardWin=3,
    #               savePath=getDataFilePath('GoBang_3_3_1'), logName='log.txt')

    # game = Gobang(Agent, showProcess=False, player=0, mode=0,
    #                       boardSize=3, boardWin=3,
    #                       savePath=getDataFilePath('GoBang_3_3_2'), logName='log.txt')
    game.run()
    # game.testAgent(getDataFilePath('GoBang_5_4_1/model_1000-1500.h5'), getDataFilePath('GoBang_5_4_1/model_0-500.h5'))


