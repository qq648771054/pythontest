from OpenAiGym.lib import *
import threading

class Memory(object):
    def __init__(self, size):
        self.size = size
        self.arr = []
        self.iter = 0

    def append(self, m):
        if len(self.arr) < self.size:
            self.arr.append(m)
        else:
            self.arr[self.iter] = m
            self.iter = (self.iter + 1) % self.size

    def sample(self, batch):
        if batch > len(self.arr):
            res = self.arr[:]
            random.shuffle(res)
            return res
        else:
            idx = np.random.choice(len(self.arr), size=batch)
            return [self.arr[x] for x in idx]

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

    def saveMemory(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self, batch=32, gamma=0.95):
        sample = self.memory.sample(batch)
        states, actions, rewards, next_states = vstack(sample)
        values = self.model.predict(np.array(states))
        next_values = self.model.predict(np.array(next_states))
        for v, nv, a, r in zip(values, next_values, actions, rewards):
            v[a] = r + gamma * nv.max()
        self.model.fit(np.array(states), values, epochs=10, verbose=0)

    def _createModel(self):
        input1 = tf.keras.Input(shape=(self.env.stateLen, ))
        output = tf.keras.layers.Dense(100, activation='relu')(input1)
        output1 = tf.keras.layers.Dense(self.env.actionLen, activation='linear')(output)
        model = tf.keras.Model(inputs=input1, outputs=output1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.losses.mse,
        )
        return model

class CartPole(Game):
    saveType = {
        'episode': (str, int, 0)
    }
    bakFrequence = 500

    actionLen = 2
    stateLen = 4

    def step(self, action):
        # next_state, reward, done, info = self.env.step(action)
        # if done:
        #     reward = -10000
        # return next_state, reward, done

        next_state, reward, done, info = self.env.step(action)
        x, x_dot, theta, theta_dot = next_state
        r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
        r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return next_state, reward, done

    def train(self, showProcess=False):
        agent = Agent(self, memorySize=300)
        episode = 0
        config = self.load()
        if config:
            episode = config['episode']
            if config['model']:
                agent.model = config['model']
        while True:
            episode += 1
            step = 0
            state = self.env.reset()
            showProcess and self.render(1)
            while True:
                action = agent.chooseAction(state)
                next_state, reward, done = self.step(action)
                agent.saveMemory(state, action, reward, next_state)
                step += 1
                state = next_state
                showProcess and self.render(0.016)
                if done:
                    break
            agent.learn()
            agent.memory.clear()
            self.log(f'episode {episode}: step {step}')

    def play(self):
        thread = threading.Thread()
        thread.start()

if __name__ == '__main__':
    cartPole = CartPole('CartPole-v1', getDataFilePath('CartPole_1'))
    cartPole.train(showProcess=False)
