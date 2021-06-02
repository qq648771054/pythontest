from lib import *
from OpenAiGym.CartPole.CartPoleBase import CartPoleBase

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
        self.modelPre = copyModel(self.model)
        self.memory = Memory(memorySize)

    def chooseAction(self, state, greedy=0.9):
        if random.random() < greedy:
            return self.model.predict(np.array([state]))[0].argmax()
        else:
            return random.randint(0, self.env.actionLen - 1)

    def saveMemory(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def syncModel(self):
        self.modelPre = copyModel(self.model)

    def learn(self, batch=32, gamma=0.95):
        sample = self.memory.sample(batch)
        states, actions, rewards, next_states = vstack(sample)
        values = self.model.predict(np.array(states))
        next_values = self.model.predict(np.array(next_states))
        next_values_p = self.modelPre.predict(np.array(next_states))
        for v, nv, nvp, a, r in zip(values, next_values, next_values_p, actions, rewards):
            idx = nv.argmax()
            v[a] = r + gamma * nvp[idx]
        self.model.fit(np.array(states), values, epochs=1, verbose=0)

    def _createModel(self):
        input1 = tf.keras.Input(shape=(self.env.stateLen, ))
        x = tf.keras.layers.Dense(30, activation='relu')(input1)
        x = tf.keras.layers.Dense(30, activation='relu')(x)
        output1 = tf.keras.layers.Dense(self.env.actionLen, activation='linear')(x)
        model = tf.keras.Model(inputs=input1, outputs=output1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.losses.mse,
        )
        return model

class CartPole(CartPoleBase):
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward = -1 if done else 0
        return next_state, reward, done

    def train(self, showProcess=False):
        startTime = time.time()
        agent = Agent(self, memorySize=1000)
        agent.model.summary()
        episode = 0
        config = self.load()
        steps = [0] * 10
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
                if step % 5 == 0:
                    agent.learn()
                showProcess and self.render(0.016)
                if done:
                    break
            agent.learn()
            agent.syncModel()
            self.log(f'episode {episode}: step {step}')
            self.save(agent.model, episode=episode)
            self.saveLog()
            steps[episode % 10] = step
            if sum(steps) >= 1600:
                break
        return time.time() - startTime, episode

'''
与main_6的主要不同:
用最新网络的动作去选择老网络的值进行更新(double dqn)
统计:
    收敛次数        时间
1     49        00:02:24
2     59        00:02:46
3     60        00:03:03
4     27        00:01:34
5     75        00:03:07
all   270       00:12:55
'''
if __name__ == '__main__':
    root = getDataFilePath(f'CartPole/CartPole_7/')
    if not os.path.exists(root):
        os.mkdir(root)
    startTime = time.time()
    totalEpisode = 0
    for i in range(5):
        cartPole = CartPole('CartPole-v0', os.path.join(root, f'CartPole_7_{i + 1}'))
        spendTime, episode = cartPole.train(showProcess=False)
        totalEpisode += episode
        print(f'train {i + 1}, spendTime {second2Str(int(spendTime))}, episode {episode}')
    print(f'testFinish spend {second2Str(int(time.time() - startTime))}, episode {totalEpisode}')
