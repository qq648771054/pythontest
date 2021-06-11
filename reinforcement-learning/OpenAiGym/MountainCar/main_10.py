from lib import *
from OpenAiGym.MountainCar.MountainCarBase import MountainCarBase
import threading
import gym

class Memory(object):
    def __init__(self, size):
        self.size = size
        self.arr = []
        self.weight = []
        self.iter = 0
        self.newly = []

    def append(self, m):
        if len(self.arr) < self.size:
            self.arr.append(m)
            self.weight.append(1)
        else:
            self.arr[self.iter] = m
            self.weight[self.iter] = 1
        self.newly.append(self.iter)
        self.iter = (self.iter + 1) % self.size

    def sample(self, batch):
        maxW = max(self.weight)
        for n in self.newly:
            self.weight[n] = maxW
        self.newly = []
        idx = sorted([i for i in range(len(self.arr))], key=lambda a: random.random())[:batch]
        return idx, [self.arr[x] for x in idx], [self.weight[x] for x in idx]

    def updateErrors(self, idx, errors):
        for x, e in zip(idx, errors):
            assert not math.isnan(e)
            self.weight[x] = abs(e) ** 0.5

    def clear(self):
        self.arr = []
        self.weight = []
        self.newly = []
        self.iter = 0

    def empty(self):
        return len(self.arr) == 0

class Agent(object):
    def __init__(self, env, memorySize=1000):
        self.env = env
        self.model = self._createModel()
        self.modelPre = copyModel(self.model)
        self.memory = Memory(memorySize)

    def chooseAction(self, state, greedy=0.9):
        if random.random() < greedy:
            return self.model.predict(np.array([state]))[0].argmax()
        else:
            return random.randint(0, self.env.actionLen - 1)

    def saveMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch=32, gamma=0.95):
        if self.memory.empty():
            return
        idxs, samples, weights = self.memory.sample(batch)
        states, actions, rewards, next_states, dones = vstack(samples)
        values = self.model.predict(np.array(states))
        next_values = self.model.predict(np.array(next_states))
        next
        errors = []
        for v, nv, a, r, d in zip(values, next_values, actions, rewards, dones):
            q = r if d else r + gamma * nv.max()
            errors.append(v[a] - q)
            v[a] = q
        self.model.fit(np.array(states), values, sample_weight=np.array(weights), epochs=1, verbose=0)
        self.memory.updateErrors(idxs, errors)

    def _createModel(self):
        input1 = tf.keras.Input(shape=(self.env.stateLen, ))
        x = tf.keras.layers.Dense(30, activation='relu')(input1)
        x = tf.keras.layers.Dense(30, activation='relu')(x)
        svalue = tf.keras.layers.Dense(1)(x)
        avalue = tf.keras.layers.Dense(self.env.actionLen)(x)
        mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(avalue)
        avalue = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([avalue, mean])
        output1 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([svalue, avalue])
        model = tf.keras.Model(inputs=input1, outputs=output1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.losses.mse,
        )
        return model

class Worker(threading.Thread):
    def __init__(self, parentAgent, game, env, idx):
        threading.Thread.__init__(self)
        self.parentAgent = parentAgent
        self.env = env
        self.game = game
        self.agent = Agent(env)
        copyModel(self.agent.model, self.parentAgent.model)
        self.idx = idx

    def run(self):
        while True:
            step = 0
            paths = []
            maxHeight = 0
            state = self.game.reset()
            while True:
                action = self.agent.chooseAction(state)
                next_state, reward, done = self.env.warpper(*self.game.step(action))
                self.parentAgent.saveMemory(state, action, reward, next_state, done)
                step += 1
                paths.append(action)
                maxHeight = max(maxHeight, abs(next_state[0] + 0.5))
                state = next_state
                if done:
                    break
            copyModel(self.agent.model, self.parentAgent.model)
            self.env.episode += 1
            self.env.log(f'agent {self.idx}, episode {self.env.episode}: step {step}, max height {maxHeight}, paths: {paths}')
            if self.env.episode % 30 == 0:
                self.env.saveModel()

class Learner(threading.Thread):
    def __init__(self, agent, env):
        threading.Thread.__init__(self)
        self.agent = agent
        self.env = env

    def run(self):
        while True:
            self.agent.learn(512)
            time.sleep(0.5)

class MountainCar(MountainCarBase):
    def warpper(self, next_state, reward, done, info):
        pos = next_state[0]
        reward = abs(pos + 0.5) ** 2.0 if pos < 0.7 else 1000
        return next_state, reward, done

    def train(self, childCnt=1):
        self.agent = Agent(self, memorySize=1500)
        self.agent.model.summary()
        config = self.load()
        self.episode = 0
        self.spendTime = 0
        self.savedTime = time.time()
        if config:
            self.episode = config['episode']
            self.spendTime = config['spendTime']
            if config['model']:
                self.agent.model = config['model']
        workers = []
        envs = []
        learner = Learner(self.agent, self)
        for i in range(childCnt):
            env = gym.make(self.gameName)
            worker = Worker(self.agent, env, self, i)
            workers.append(worker)
            envs.append(env)

        learner.start()
        for worker in workers:
            worker.start()

        learner.join()
        for worker in workers:
            worker.join()

    def display(self):
        self.agent = Agent(self)
        config = self.load()
        if config:
            if config['model']:
                self.agent.model = config['model']
        episode = 0
        while True:
            step = 0
            paths = []
            state = self.env.reset()
            episode += 1
            self.render(0.5)
            while True:
                action = self.agent.chooseAction(state)
                next_state, reward, done = self.warpper(*self.env.step(action))
                step += 1
                paths.append(action)
                state = next_state
                self.render(0.016)
                if done:
                    break
            episode += 1
            print(f'episode {episode}: step {step}, paths: {paths}')

    def saveModel(self):
        self.spendTime += time.time() - self.savedTime
        self.savedTime = time.time()
        self.save(self.agent.model, episode=self.episode, spendTime=self.spendTime)
        self.saveLog()

'''
与main_21主要不同:
改变reward算法,变成得分
'''
if __name__ == '__main__':
    root = getDataFilePath(f'MountainCar/MountainCar_10/')
    if not os.path.exists(root):
        os.mkdir(root)
    game = MountainCar('MountainCar-v0', os.path.join(root, f'MountainCar_10_1'))
    game.train()
    # game.display()
