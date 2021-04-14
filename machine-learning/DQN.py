from Lib import *
import numpy as np
import pandas as pd
import time
import tkinter as tk
import threading
import tensorflow as tf
import gym
import copy


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus= tf.config.list_physical_devices('GPU')
if len(gpus) > 0: tf.config.experimental.set_memory_growth(gpus[0], True)

class Agent_NN(object):
    def __init__(self, env, model, reward_decay=0.9, e_greedy=0.9):
        self.env = env
        self.model = model
        self.reward_decay, self.e_greddy = reward_decay, e_greedy
        self.memory = []
        self.memory_size = 1000
        self.memory_iter = 0
        self.batch_size = 32

    def save_exp(self, state, action, reward, next_state):
        if len(self.memory) < self.memory_size:
            self.memory.append((state, action, reward, next_state))
        else:
            self.memory[self.memory_iter] = state, action, reward, next_state
            self.memory_iter = (self.memory_iter + 1) % self.memory_size

    def learn(self):
        raise NotImplementedError

    def choose_action(self, state):
        if np.random.rand() >= self.e_greddy:
            return np.random.randint(0, self.env.actionLen)
        else:
            act_values = self.model.predict(addAixs(state))
            return np.argmax(act_values[0])

class Agent_DQN(Agent_NN):
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batches = np.random.choice(len(self.memory), self.batch_size)
        xs, ys = [], []
        for i in batches:
            state, action, reward, next_state = self.memory[i]
            q_target = reward + self.reward_decay * self.model.predict(addAixs(next_state))[0].max()
            q_predict = self.model.predict(addAixs(state))
            q_predict[0][action] = q_target
            xs.append(state)
            ys.append(q_predict[0])
        self.model.fit(np.array(xs), np.array(ys), epochs=1, verbose=0)

class EnvNN(object):
    def __init__(self, agentType):
        self.agent = agentType(self, self._buildModel())

    @property
    def actionLen(self):
        raise NotImplementedError

    @property
    def stateShape(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def _buildModel(self):
        raise NotImplementedError

class EnvOpenAI(EnvNN):
    def __init__(self, env, agentType):
        super(EnvOpenAI, self).__init__(agentType)
        self._env = env

    @property
    def actionLen(self):
        return self._env.action_space.n

    @property
    def stateShape(self):
        return self._env.observation_space.shape

    def reset(self):
        return self._env.reset()

    def render(self):
        self._env.render()

    def step(self, action):
        return self._env.step(action)

class CartPole_v0(EnvOpenAI):
    def _buildModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation='relu', input_shape=self.stateShape),
            tf.keras.layers.Dense(self.actionLen, activation='linear')
        ])
        return model

class EnvTk(EnvNN, tk.Tk):
    ACTION = []

    def __init__(self, map, agentType):
        tk.Tk.__init__(self)
        self._buildMap(map)
        EnvNN.__init__(self, agentType)

    @property
    def actionLen(self):
        return len(self.ACTION)

    def _buildMap(self, map):
        raise NotImplementedError

    def render(self, sleepTime=0.5):
        self.update()
        time.sleep(sleepTime)

class Maze(EnvTk):
    class TYPE:
        GROUND = 0
        START = 1
        END = 2
        TRAP = 3
    TYPE2ID = {
        '.': TYPE.GROUND,
        's': TYPE.START,
        'e': TYPE.END,
        '#': TYPE.TRAP,
    }
    TYPE2REWARD = {
        TYPE.GROUND: 0,
        TYPE.START: 0,
        TYPE.END: 1,
        TYPE.TRAP: -1
    }
    TYPE2DONE = {
        TYPE.GROUND: False,
        TYPE.START: False,
        TYPE.END: True,
        TYPE.TRAP: True
    }
    TYPE2COLOR = {
        TYPE.GROUND: 'white',
        TYPE.START: 'black',
        TYPE.END: 'green',
        TYPE.TRAP: 'red'
    }

    ACTION = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    MAZE_W, MAZE_H = 30, 30
    GRID_W, GRID_H = 25, 25

    @property
    def stateShape(self):
        return self.height, self.width

    def _buildMap(self, map):
        map = [m.strip() for m in map.split('\n') if len(m.strip()) > 0]
        self.rawMap = map
        self.width = len(map[0])
        self.height = len(map)
        self.map = []
        for i in range(self.height):
            self.map.append([])
            for j in range(self.width):
                self.map[i].append(self.TYPE2ID[map[i][j]])
                if self.map[i][j] == self.TYPE.START:
                    self.x, self.y = j, i
        self._createMap()

    def reset(self):
        map = self.rawMap
        self.map = []
        for i in range(self.height):
            self.map.append([])
            for j in range(self.width):
                self.map[i].append(self.TYPE2ID[map[i][j]])
                if self.map[i][j] == self.TYPE.START:
                    self.x, self.y = j, i
        self._createMap()
        return np.array(self.map)

    def step(self, action):
        state = np.array(self.map)
        x, y = self.x + self.ACTION[action][0], self.y + self.ACTION[action][1]
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return state, action, -1, np.array(self.map), True
        else:
            nextPath = self.map[y][x]
            self.map[y][x], self.map[self.y][self.x] = self.TYPE.START, self.TYPE.GROUND
            self._updateGrid(x, y)
            self._updateGrid(self.x, self.y)
            self.y, self.x = y, x
            reward, done = self.TYPE2REWARD[nextPath], self.TYPE2DONE[nextPath]
            return state, action, reward, np.array(self.map), done

    def _buildModel(self, learning_rate=0.01):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.stateShape),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(self.actionLen, activation='linear')
        ])
        model.compile(
            loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate)
        )
        return model

    def _createMap(self):
        if not hasattr(self, 'canvas'):
            self.canvas = tk.Canvas(self, bg='white', height=self.MAZE_H * self.height, width=self.MAZE_W * self.width)
            # create lines
            right, bottom = self.MAZE_W * self.width, self.MAZE_H * self.height
            for c in range(0, right, self.MAZE_W):
                self.canvas.create_line(c, 0, c, bottom)
            for r in range(0, bottom, self.MAZE_H):
                self.canvas.create_line(0, r, right, r)
            self.canvas.pack()
        # create grids
        self.grids = []
        for i in range(self.height):
            self.grids.append([0] * self.width)
            for j in range(self.width):
                self._updateGrid(j, i)

    def _updateGrid(self, x, y):
        if self.grids[y][x]:
            self.canvas.delete(self.grids[y][x])
        self.grids[y][x] = self.canvas.create_rectangle(
            (x + 0.5) * self.MAZE_W - 0.5 * self.GRID_W, (y + 0.5) * self.MAZE_H - 0.5 * self.GRID_H,
            (x + 0.5) * self.MAZE_W + 0.5 * self.GRID_W, (y + 0.5) * self.MAZE_H + 0.5 * self.GRID_H,
            fill=self.TYPE2COLOR[self.map[y][x]]
        )

class ThreadBase(threading.Thread):
    def __init__(self, showProcess=True, savePath='', **kwargs):
        threading.Thread.__init__(self)
        self.showProcess = showProcess
        self.savePath = savePath
        self.args = kwargs

    def run(self):
        self.createEnv()
        if self.savepath and os.path.exists(self.savepath):
            self.env.agent.model = tf.keras.models.load_model(self.savepath)
        episode = 0
        while True:
            state = self.env.reset()
            self.showProcess and self.env.render()
            episode += 1
            startTime = time.time()
            step = 0
            while True:
                action = self.env.agent.choose_action(state)
                state, action, reward, next_state, done = self.env.step(action)
                self.env.agent.save_exp(state, action, reward, next_state)
                step += 1
                state = next_state
                self.showProcess and self.env.render(0.05)
                if done:
                    break
            print('episode {}, result {}, takes {} steps {} second'.format(episode, reward == 1, step, time.time() - startTime))
            self.env.agent.learn()
            if self.savepath:
                self.env.agent.model.save(self.savepath)

    def createEnv(self):
        raise NotImplementedError

class ThreadMaze(ThreadBase):
    def createEnv(self):
        self.env = Maze(readFile(getDataFilePath('DQN_SimpleMaze.txt')), Agent_DQN)

if __name__ == '__main__':
    thread = ThreadMaze(showProcess=False, savePath=getDataFilePath('dqn.h4'))
    thread.start()
    while True:
        cmd = input()
        if cmd == '0':
            thread.showProcess = False
        else:
            thread.showProcess = True
