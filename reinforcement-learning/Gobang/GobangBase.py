from lib import *
from Gobang.Game import Game

class GobangBase(object):
    MODE_GREEDY = 1
    MODE_RANDOM = 2
    saveType = {
        'episode': (str, int, 0),
        'spendTime': (str, float, 0)
    }
    bakFrequence = 500

    def __init__(self, size=15, winLength=5, savePath=''):
        self.env = Game(size, winLength, onGridClick=self.onGridClick)
        self.savePath = savePath
        if self.savePath:
            self.logPath = os.path.join(self.savePath, 'log.txt')
            self.logTempPath = os.path.join(self.savePath, 'temp_log.txt')
            self.configPath = os.path.join(self.savePath, 'config.txt')
            if not os.path.exists(self.savePath):
                os.mkdir(self.savePath)
                f = open(self.logPath, 'w')
                f.close()
                f = open(self.logTempPath, 'w')
                f.close()
                f = open(self.configPath, 'w')
                f.close()
            self.logs = []
        self.lastClick = None

        self.stateSize = (self.env.size, self.env.size, 1)
        self.actionSize = self.env.actionSize

    def __getattr__(self, item):
        return getattr(self.env, item)

    def train(self, *args, **kwargs):
        pass

    def play(self, player, agentType):
        agent = agentType(self, 0)
        config = self.load()
        if config is not None:
            if config['model']:
                agent.model = config['model']
        agent.model.summary()
        while True:
            state = self.env.reset()
            self.render(0.5)
            path = []
            currentPlayer = 1
            while True:
                if currentPlayer == player:
                    action = self.waitPlayerClick()
                    next_state, p, winner = self.env.step(action)
                else:
                    action = agent.chooseAction(state, GobangBase.MODE_GREEDY)
                    next_state, p, winner = self.env.step(action)
                state = next_state
                currentPlayer = 3 - currentPlayer
                path.append(action)
                self.render(0.5)
                if winner is not None:
                    break
            print(f'path {path}, winner {winner}')

    def onGridClick(self, x, y):
        self.lastClick = y * self.env.size + x

    def waitPlayerClick(self):
        validActions = self.env.validActions()
        while True:
            self.lastClick = None
            self.render()
            time.sleep(0.015)
            if self.lastClick is not None and self.lastClick in validActions:
                return self.lastClick

    def load(self):
        config = None
        if self.savePath:
            config = self.readConfig()
            modelPath = os.path.join(self.savePath, 'model_{}-{}.h5'.format(
                (config['episode'] // self.bakFrequence) * self.bakFrequence,
                ((config['episode'] // self.bakFrequence) + 1) * self.bakFrequence,
            ))
            if os.path.exists(modelPath):
                model = tf.keras.models.load_model(modelPath)
                config['model'] = model
            else:
                config['model'] = None
        return config

    def readConfig(self):
        config = {}
        with open(self.configPath) as f:
            line = f.readline().strip()
            while line:
                idx = line.find('=')
                if idx != -1:
                    name = line[:idx].strip()
                    data = line[idx + 1:].strip()
                    config[name] = self.saveType[name][1](data)
                line = f.readline()
        for k in self.saveType:
            if k not in config:
                config[k] = self.saveType[k][2]
        return config

    def save(self, model, episode, **kwargs):
        if self.savePath:
            modelPath = os.path.join(self.savePath, 'model_{}-{}.h5'.format(
                (episode // self.bakFrequence) * self.bakFrequence,
                ((episode // self.bakFrequence) + 1) * self.bakFrequence,
            ))
            model.save(modelPath)
            kwargs['episode'] = episode
            with open(self.configPath, 'w') as f:
                for k, v in kwargs.items():
                    f.write('{}={}\n'.format(k, self.saveType[k][0](v)))
            return tf.keras.models.load_model(modelPath)
        else:
            return copyModel(model)

    def log(self, s):
        currentTime = str(datetime.datetime.now())
        print(currentTime, ':', s)
        if self.savePath:
            self.logs.append(currentTime + ':')
            self.logs.append(s + '\n')

    def saveLog(self, temp=False):
        if len(self.logs) > 0:
            logFileTemp = open(self.logTempPath, 'w')
            if temp:
                logFileTemp.write(''.join(self.logs))
                logFileTemp.flush()
            else:
                logFile = open(self.logPath, 'a')
                logFile.write(''.join(self.logs))
                logFile.flush()
                logFile.close()
                self.logs = []
            logFileTemp.close()

    def render(self, sleepTime=None):
        self.env.render()
        if sleepTime is not None:
            time.sleep(sleepTime)
