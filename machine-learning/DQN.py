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

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def _buildModel(self):
        raise NotImplementedError

class EnvOpenAI(EnvNN):
    def __init__(self, env, agentType):
        self._env = env
        super(EnvOpenAI, self).__init__(agentType)

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
        return self._env.step(action)[: 3]

class CartPole_v0(EnvOpenAI):
    def _buildModel(self, learning_rate=0.01):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation='relu', input_shape=self.stateShape),
            tf.keras.layers.Dense(self.actionLen, activation='linear')
        ])
        model.compile(
            loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate)
        )
        return model

    def step(self, action):
        next_state, reward, done = super(CartPole_v0, self).step(action)
        x, x_dot, theta, theta_dot = next_state
        env = self._env
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return next_state, reward, done

class EnvTk(EnvNN, tk.Tk):
    ACTION = []

    def __init__(self, agentType):
        tk.Tk.__init__(self)
        EnvNN.__init__(self, agentType)

    @property
    def actionLen(self):
        return len(self.ACTION)

    def render(self):
        self.update()

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

    def __init__(self, width, height, agentType):
        self.width, self.height = width, height
        super(Maze, self).__init__(agentType)

    @property
    def stateShape(self):
        return self.height, self.width

    def reset(self, mapData):
        map = [m.strip() for m in mapData.split('\n') if len(m.strip()) > 0]
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
        x, y = self.x + self.ACTION[action][0], self.y + self.ACTION[action][1]
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return np.array(self.map), -1, True
        else:
            nextPath = self.map[y][x]
            self.map[y][x], self.map[self.y][self.x] = self.TYPE.START, self.TYPE.GROUND
            self._updateGrid(x, y)
            self._updateGrid(self.x, self.y)
            self.y, self.x = y, x
            reward, done = self.TYPE2REWARD[nextPath], self.TYPE2DONE[nextPath]
            return np.array(self.map), reward, done

    def _buildModel(self, learning_rate=0.01):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.stateShape),
            tf.keras.layers.Dense(128, activation='relu'),
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

    def loadModel(self, agent):
        if self.savePath and os.path.exists(self.savePath):
            agent.model = tf.keras.models.load_model(self.savePath)

    def saveModel(self, agent):
        if self.savePath:
            agent.model.save(self.savePath)

    def render(self, env, sleepTime=None):
        if self.showProcess:
            env.render()
            if sleepTime:
                time.sleep(sleepTime)

class ThreadMaze(ThreadBase):
    WIDTH, HEIGHT = 5, 5
    def run(self):
        mapDatas = []
        mapIter = 0
        dir = getDataFilePath('DQN_Maze')
        for file in os.listdir(dir):
            file = os.path.join(dir, file)
            mapDatas.append(readFile(file))
        env = Maze(self.WIDTH, self.WIDTH, Agent_DQN)
        agent = env.agent
        self.loadModel(agent)
        episode = 0
        record = [0] * 50
        success = 0
        while True:
            state = env.reset(mapDatas[mapIter])
            self.render(env)
            episode += 1
            startTime = time.time()
            step = 0
            while True:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.save_exp(state, action, reward, next_state)
                step += 1
                state = next_state
                self.render(env, 0.05)
                if done:
                    break
            isSuccess = int(reward == 1)
            print('episode {}, mapIter {}, result {}, takes {} steps {} second'.format(episode, mapIter, isSuccess, step, time.time() - startTime))
            success += isSuccess - record[episode % 50]
            record[episode % 50] = isSuccess
            agent.learn()
            self.saveModel(agent)
            if success >= 30:
                record = [0] * 50
                success = 0
                mapIter = (mapIter + 1) % len(mapDatas)
                print('switch mapIter to {}'.format(mapIter))

class ThreadCartPole(ThreadBase):
    def run(self):
        env = CartPole_v0(gym.make('CartPole-v0'), Agent_DQN)
        agent = env.agent
        self.loadModel(agent)
        episode = 0
        while True:
            state = env.reset()
            self.render(env, 0.5)
            episode += 1
            step = 0
            while True:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.save_exp(state, action, reward, next_state)
                step += 1
                state = next_state
                self.render(env)
                if done:
                    break
            agent.learn()
            print('episode {}, steps {}'.format(episode, step))
            self.saveModel(agent)

if __name__ == '__main__':
    # thread = ThreadMaze(showProcess=False, savePath=getDataFilePath('dqn_maze_record.h5'))
    thread = ThreadCartPole(showProcess=True, savePath=getDataFilePath('dqn_cartPole_record.h5'))
    thread.start()
    while True:
        cmd = input()
        if cmd == '0':
            thread.showProcess = False
        else:
            thread.showProcess = True

