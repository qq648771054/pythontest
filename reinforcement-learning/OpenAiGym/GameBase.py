from lib import *
import gym


class GameBase(object):
    saveType = {
        'episode': (str, int, 0)
    }
    bakFrequence = 500

    def __init__(self, name, savePath=''):
        self.env = gym.make(name)
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

    def __getattr__(self, item):
        return getattr(self.env, item)

    def train(self, *args, **kwargs):
        pass

    def play(self, *args, **kwargs):
        self.env.reset()
        self.env.render()
        self.env.unwrapped.viewer.window.on_key_press = self.on_key_press
        self.env.unwrapped.viewer.window.on_mouse_motion = self.on_mouse_motion
        self.env.unwrapped.viewer.window.on_mouse_press = self.on_mouse_press

    def on_key_press(self, key, mode):
        pass

    def on_mouse_motion(self, x, y, dx, dy):
        pass

    def on_mouse_press(self, x, y, button, modifier):
        pass

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


class KEY:
    A = 97
    B = 98
    C = 99
    D = 100
    E = 101
    F = 102
    G = 103
    H = 104
    I = 105
    J = 106
    K = 107
    L = 108
    M = 109
    N = 110
    O = 111
    P = 112
    Q = 113
    R = 114
    S = 115
    T = 116
    U = 117
    V = 118
    W = 119
    X = 120
    Y = 121
    Z = 122
    LEFT = 65361
    RIGHT = 65363
    UP = 65362
    DOWN = 65364


class MOUSE:
    LEFT = 1
    SCROLL = 2
    RIGHT = 4

class MOD:
    SHIFT = 1
    CTRL = 2
    ALT = 4
    CAPSLOCK = 8
    NUMLOCK = 16
    WINDOWS = 32
    SCROLLLOCK = 256
