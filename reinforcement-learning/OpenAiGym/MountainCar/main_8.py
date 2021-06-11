from lib import *
from OpenAiGym.MountainCar.MountainCarBase import MountainCarBase
import threading
import gym

class Memory(object):
    def __init__(self):
        self.arr = []

    def append(self, m):
        self.arr.append(m)

    def sample(self):
        return self.arr[:]

    def clear(self):
        self.arr = []

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.model = self._createModel()
        self.memory = Memory()

    def chooseAction(self, state):
        prop = self.model.predict(np.array([state]))[0]
        return np.random.choice(self.env.actionLen, p=prop)

    def saveMemory(self, state, action, reward):
        self.memory.append((state, action, reward))

    def popMemory(self):
        m = self.memory.sample()
        self.memory.clear()
        return m

    def learn(self, sample, gamma=0.95):
        states, actions, rewards = vstack(sample)
        actions = [[1.0 if i == a else 0.0 for i in range(self.env.actionLen)] for a in actions]
        self.model.fit(np.array(states), np.array(actions), sample_weight=self.discount_rewards(rewards, gamma), epochs=5, verbose=0)

    def discount_rewards(self, rewards, gamma):
        prior = 0
        out = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            prior = prior * gamma + rewards[i]
            out[i] = prior
        return out / np.std(out - np.mean(out))

    def _createModel(self, learning_rate=0.001):
        input1 = tf.keras.Input(shape=(self.env.stateLen, ))
        x = tf.keras.layers.Dense(100, activation='relu')(input1)
        x = tf.keras.layers.Dropout(0.1)(x)
        output1 = tf.keras.layers.Dense(self.env.actionLen, activation='softmax')(x)
        model = tf.keras.Model(inputs=input1, outputs=output1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.losses.categorical_crossentropy,
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
                self.agent.saveMemory(state, action, reward)
                step += 1
                paths.append(action)
                maxHeight = max(maxHeight, abs(next_state[0] + 0.5))
                state = next_state
                if done:
                    break
            self.parentAgent.learn(self.agent.popMemory())
            copyModel(self.agent.model, self.parentAgent.model)
            self.env.episode += 1
            self.env.log(f'agent {self.idx}, episode {self.env.episode}: step {step}, max height {maxHeight}, paths: {paths}')
            if self.env.episode % 30 == 0:
                self.env.saveModel()

class MountainCar(MountainCarBase):
    def warpper(self, next_state, reward, done, info):
        # reward = abs(next_state[0] + 0.5) ** 2.0
        reward = abs(next_state[0] + 0.5)
        return next_state, reward, done

    def train(self, childCnt=8):
        self.agent = Agent(self)
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
        for i in range(childCnt):
            env = gym.make(self.gameName)
            worker = Worker(self.agent, env, self, i)
            workers.append(worker)
            envs.append(env)

        for worker in workers:
            worker.start()

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
使用A3C+policy gradient
'''
if __name__ == '__main__':
    root = getDataFilePath(f'MountainCar/MountainCar_8/')
    if not os.path.exists(root):
        os.mkdir(root)
    game = MountainCar('MountainCar-v0', os.path.join(root, f'MountainCar_8_1'))
    game.train()
    # game.display()
