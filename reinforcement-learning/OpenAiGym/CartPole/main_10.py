from lib import *
from OpenAiGym.CartPole.CartPoleBase import CartPoleBase

class Memory(object):
    def __init__(self, size):
        self.size = size
        self.arr = []
        self.weight = []
        self.iter = 0
        self.maxWeight = 1.0

    def append(self, m):
        if len(self.arr) < self.size:
            self.arr.append(m)
            self.weight.append(self.maxWeight)
        else:
            self.arr[self.iter] = m
            self.weight[self.iter] = self.maxWeight
            self.iter = (self.iter + 1) % self.size

    def sample(self, batch):
        idx = np.random.choice(len(self.arr), size=batch, p=normallize(self.weight))
        return idx, [self.arr[x] for x in idx]

    def updateErrors(self, idx, errors):
        for x, e in zip(idx, errors):
            self.weight[x] = min(self.maxWeight, abs(e))

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
        idxs, sample = self.memory.sample(batch)
        states, actions, rewards, next_states = vstack(sample)
        values = self.model.predict(np.array(states))
        next_values = self.model.predict(np.array(next_states))
        next_values_p = self.modelPre.predict(np.array(next_states))
        errors = []
        for v, nv, nvp, a, r in zip(values, next_values, next_values_p, actions, rewards):
            idx = nv.argmax()
            q = r + gamma * nvp[idx]
            errors.append(v[a] - q)
            v[a] = q
        self.model.fit(np.array(states), values, epochs=1, verbose=0)
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

class CartPole(CartPoleBase):
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward = -1 if done else 0
        return next_state, reward, done

    def train(self, showProcess=False):
        agent = Agent(self, memorySize=1000)
        agent.model.summary()
        spendTime = 0
        episode = 0
        steps = [0] * 10
        config = self.load()
        if config:
            episode = config['episode']
            spendTime = spendTime
            if config['model']:
                agent.model = config['model']
        while True:
            startTime = time.time()
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
            self.save(agent.model, episode=episode, spendTime=spendTime)
            self.saveLog()
            steps[episode % 10] = step
            spendTime += time.time() - startTime
            if sum(steps) >= 1600:
                break
        return spendTime, episode

'''
与main_9的主要不同:
改变网络结构,变成avalue和svalue的结构(dueling dqn)
统计:
    收敛次数        时间
1     40        00:01:45
2     56        00:02:20
3     35        00:01:27
4     43        00:02:46
5     44        00:02:30
all   218       00:10:50
'''
if __name__ == '__main__':
    root = getDataFilePath(f'CartPole/CartPole_10/')
    if not os.path.exists(root):
        os.mkdir(root)
    startTime = time.time()
    totalEpisode = 0
    for i in range(5):
        cartPole = CartPole('CartPole-v0', os.path.join(root, f'CartPole_10_{i + 1}'))
        spendTime, episode = cartPole.train(showProcess=False)
        totalEpisode += episode
        print(f'train {i + 1}, spendTime {second2Str(int(spendTime))}, episode {episode}')
    print(f'testFinish spend {second2Str(int(time.time() - startTime))}, episode {totalEpisode}')
