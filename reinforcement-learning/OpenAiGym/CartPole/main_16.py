from lib import *
from OpenAiGym.CartPole.CartPoleBase import CartPoleBase

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

    def learn(self, gamma=0.95):
        sample = self.memory.sample()
        states, actions, rewards = vstack(sample)
        # actions = [[1.0 if i == a else 0.0 for i in range(self.env.actionLen)] for a in actions] * self.model.predict(np.array(states))
        actions = [[1.0 if i == a else 0.0 for i in range(self.env.actionLen)] for a in actions]
        self.model.fit(np.array(states), np.array(actions), sample_weight=self.discount_rewards(rewards, gamma), epochs=5, verbose=0)
        self.memory.clear()

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

class CartPole(CartPoleBase):
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward = -1 if done else 0.01
        return next_state, reward, done

    def train(self, showProcess=False):
        agent = Agent(self)
        agent.model.summary()
        spendTime = 0
        episode = 0
        steps = [0] * 10
        config = self.load()
        if config:
            episode = config['episode']
            spendTime = config['spendTime']
            if config['model']:
                agent.model = config['model']
        while True:
            startTime = time.time()
            episode += 1
            step = 0
            paths = []
            state = self.env.reset()
            showProcess and self.render(1)
            while True:
                action = agent.chooseAction(state)
                next_state, reward, done = self.step(action)
                agent.saveMemory(state, action, reward)
                step += 1
                paths.append(action)
                state = next_state
                showProcess and self.render(0.016)
                if done:
                    break
            agent.learn()
            self.log(f'episode {episode}: step {step}, paths: {paths}')
            self.save(agent.model, episode=episode, spendTime=spendTime)
            self.saveLog()
            steps[episode % 10] = step
            spendTime += time.time() - startTime
            if sum(steps) >= 1600:
                break
        return spendTime, episode

'''
与main_15主要不同:
并改变reward算法
统计:
    收敛次数        时间
1     181       00:05:59
2     213       00:07:14
3     83        00:02:31
4     73        00:02:07
5     103       00:03:06
all   653       00:20:57
'''
if __name__ == '__main__':
    root = getDataFilePath(f'CartPole/CartPole_16/')
    if not os.path.exists(root):
        os.mkdir(root)
    startTime = time.time()
    totalEpisode = 0
    for i in range(5):
        cartPole = CartPole('CartPole-v0', os.path.join(root, f'CartPole_17_{i + 1}'))
        spendTime, episode = cartPole.train(showProcess=False)
        totalEpisode += episode
        print(f'train {i + 1}, spendTime {second2Str(int(spendTime))}, episode {episode}')
    print(f'testFinish spend {second2Str(int(time.time() - startTime))}, episode {totalEpisode}')
