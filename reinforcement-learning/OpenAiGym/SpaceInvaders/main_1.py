from lib import *
from OpenAiGym.SpaceInvaders.SpaceInvadersBase import SpaceInvadersBase

class Memory(object):
    def __init__(self, maxSize):
        self.maxSize = maxSize
        self.arr = []
        self.iter = 0

    def append(self, m):
        if len(self.arr) < self.maxSize:
            self.arr.append(m)
        else:
            self.arr[self.iter] = m
            self.iter = (self.iter + 1) % self.maxSize

    def sample(self, batch):
        if batch > len(self.arr):
            batch = len(self.arr)
        return random.sample(self.arr, batch)

    def size(self):
        return len(self.arr)

    def clear(self):
        self.arr = []
        self.iter = 0

class Agent(object):
    def __init__(self, env, memorySize=1000):
        self.env = env
        self.model = self._createModel()
        self.memory = Memory(memorySize)

    def chooseAction(self, state, greedy=0.9):
        if random.random() < greedy:
            return self.model.predict(np.array([state]))[0].argmax()
        else:
            return random.randint(0, self.env.actionLen - 1)

    def saveMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch=32, gamma=0.95):
        sample = self.memory.sample(batch)
        states, actions, rewards, next_states, dones = vstack(sample)
        values = self.model.predict(np.array(states))
        next_values = self.model.predict(np.array(next_states))
        for v, nv, a, r, d in zip(values, next_values, actions, rewards, dones):
            qp = r if d else r + gamma * nv.max()
            v[a] = qp
        self.model.fit(np.array(states), values, epochs=1, verbose=0)

    def _createModel(self):
        input1 = tf.keras.Input(shape=self.env.stateSize)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(input1)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        output1 = tf.keras.layers.Dense(self.env.actionLen, activation='linear')(x)
        model = tf.keras.Model(inputs=input1, outputs=output1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.losses.mse,
        )
        return model

class Game(SpaceInvadersBase):
    stateSize = (210, 160, 1)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if info['ale.lives'] == 0:
            reward = -100
        return next_state, reward, done

    def train(self, maxSize=2048):
        agent = Agent(self, memorySize=maxSize)
        agent.model.summary()
        spendTime = 0
        episode = 0
        totalStep = 0
        config = self.load()
        if config:
            episode = config['episode']
            totalStep = config['totalStep']
            spendTime = config['spendTime']
            if config['model']:
                agent.model = config['model']
        greedy = 1.0
        while True:
            startTime = time.time()
            step = 0
            rewards = 0
            state = rgb2Gray(self.env.reset())
            actions = []
            while True:
                self.render()
                action = agent.chooseAction(state, greedy)
                next_state, reward, done = self.step(action)
                next_state = rgb2Gray(next_state)
                agent.saveMemory(state, action, reward, next_state, done)
                actions.append(action)
                rewards += reward
                step += 1
                totalStep += 1
                state = next_state
                if done:
                    agent.learn(64)
                    episode += 1
                    spendTime += time.time() - startTime
                    self.log(f'episode: {episode}, step: {step}, totalStep: {totalStep}, rewards: {round(rewards + 100)}, actions {actions}')
                    # self.save(agent.model, episode=episode, spendTime=spendTime, totalStep=totalStep)
                    self.saveLog()
                    break

    def display(self):
        agent = Agent(self)
        agent.model.summary()
        config = self.load()
        if config and config['model']:
            agent.model = config['model']
        while True:
            step = 0
            rewards = 0
            state = rgb2Gray(self.env.reset())
            preState = state
            s0 = np.vstack((preState, state))
            while True:
                self.render()
                action = agent.chooseAction(s0)
                next_state, reward, done = self.step(action)
                next_state = rgb2Gray(next_state)
                s1 = np.vstack((state, next_state))
                rewards += reward
                step += 1
                state = next_state
                s0 = s1
                if done:
                    break

'''
使用dqn
只知道发射子弹,不会走
'''
if __name__ == '__main__':
    root = getDataFilePath(f'SpaceInvaders/SpaceInvaders_1/')
    if not os.path.exists(root):
        os.mkdir(root)
    game = Game('SpaceInvaders-v0', os.path.join(root, f'SpaceInvaders_1_5'))
    game.train()
    # game.display()
