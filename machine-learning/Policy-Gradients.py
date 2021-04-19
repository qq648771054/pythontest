from lib import *
import tkinter as tk
import threading
import gym


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus= tf.config.list_physical_devices('GPU')
if len(gpus) > 0: tf.config.experimental.set_memory_growth(gpus[0], True)

class Agent_Policy_Gradients(object):
    def __init__(self, env):
        self.env = env
        self.model = self._buildModel(self.env.stateShape, self.env.actionLen)
        self._ss = []
        self._as = []
        self._rs = []

    def save_exp(self, state, action, reward, next_state, done):
        self._ss.append(state)
        self._as.append(action)
        self._rs.append(reward)

    def learn(self, reward_decay=0.9):
        self.model.fit(np.array(self._ss), np.array(self._as),
                       sample_weight=self._discount_rewards(reward_decay),
                       verbose=0)
        self._ss = []
        self._as = []
        self._rs = []

    def choose_action(self, state):
        prop = self.model.predict(addAixs(state))[0]
        return np.random.choice(self.env.actionLen, p=prop)

    def _discount_rewards(self, reward_decay=0.9):
        """计算衰减reward的累加期望，并中心化和标准化处理"""
        prior = 0
        out = np.zeros_like(self._rs)
        for i in range(len(self._rs) - 1, -1, -1):
            prior = prior * reward_decay + self._rs[i]
            out[i] = prior
        return out / np.std(out - np.mean(out))

    def _buildModel(self, stateShape, actionLen, learning_rate=0.001):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=stateShape),
            # tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(actionLen, activation='softmax')
        ])
        model.compile(
            loss=tf.losses.mse,
            optimizer=tf.optimizers.RMSprop(learning_rate)
        )
        return model

class EnvNN(object):
    def __init__(self, agentType):
        self.agent = agentType(self)

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

class ThreadCartPole(ThreadBase):
    def run(self):
        env = CartPole_v0(gym.make('CartPole-v0'), Agent_Policy_Gradients)
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
                agent.save_exp(state, action, reward, next_state, done)
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

