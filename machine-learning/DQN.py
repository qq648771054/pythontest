from lib import *
import tkinter as tk
import threading
import gym


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus= tf.config.list_physical_devices('GPU')
if len(gpus) > 0: tf.config.experimental.set_memory_growth(gpus[0], True)

class Agent_NN(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def learn(self, reward_decay=0.9):
        raise NotImplementedError

    def choose_action(self, state, e_greddy=0.9):
        if np.random.rand() >= e_greddy:
            return np.random.randint(0, self.env.actionLen)
        else:
            act_values = self.model.predict(addAixs(state))
            return np.argmax(act_values[0])

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.object)
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Agent_DQN(Agent_NN):
    def __init__(self, *args, **kwargs):
        super(Agent_DQN, self).__init__(*args, **kwargs)
        self.bak_model = copyModel(self.model)
        self.learnTime = 0
        self.learnCycle = 10
        self.memory_size = 1000
        self.memory = Memory(self.memory_size)

    def save_exp(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def learn(self, reward_decay=0.9, batch_size=32):
        self.learnTime += 1
        if self.learnTime % self.learnCycle == 0:
            self.bak_model = copyModel(self.model)
        idxs, memorys = self.memory.sample(batch_size)
        xs, ys, deltas = [], [], []
        for i, memory in zip(idxs, memorys):
            state, action, reward, next_state, done = memory
            if done:
                q_target = reward
            else:
                idx = self.model.predict(addAixs(next_state))[0].argmax()
                q_target = reward + reward_decay * self.bak_model.predict(addAixs(next_state))[0][idx]
            q_predict = self.model.predict(addAixs(state))
            deltas.append(abs(q_target - q_predict[0][action]))
            q_predict[0][action] = q_target
            xs.append(state)
            ys.append(q_predict[0])
        self.model.fit(np.array(xs), np.array(ys), epochs=1, verbose=0)
        self.memory.batch_update(idxs, np.array(deltas))

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
    thread = ThreadCartPole(showProcess=True, savePath=getDataFilePath('dqn_cartPole_record1.h5'))
    thread.start()
    while True:
        cmd = input()
        if cmd == '0':
            thread.showProcess = False
        else:
            thread.showProcess = True

