from lib import *
from OpenAiGym.CartPole.CartPoleBase import CartPoleBase

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

class Agent(object):
    def __init__(self, env, memorySize=1000):
        self.env = env
        self.actor, self.critic = self._createModel()
        self.memory = Memory(memorySize)

    def chooseAction(self, state):
        prop = self.actor.predict(np.array([state]))[0]
        return np.random.choice(self.env.actionLen, p=prop)

    def save_exp(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self, batch=32, gamma=0.95):
        idx, sample, weight = self.memory.sample(batch)
        states, actions, rewards, next_states = vstack(sample)
        states, next_states = np.array(states), np.array(next_states)
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)
        v_predicts = [r + gamma * v[0] for r, v in zip(rewards, next_values)]
        td_errors = [vp - v[0] for vp, v in zip(v_predicts, values)]
        actions = [[1 if i == a else 0 for i in range(self.env.actionLen)] for a in actions]
        self.critic.fit(states, np.array(v_predicts), epochs=1, verbose=0)
        self.actor.fit(states, np.array(actions), sample_weight=np.array([e * w for e, w, in zip(td_errors, weight)]), epochs=1, verbose=0)
        self.memory.updateErrors(idx, td_errors)

    def _createModel(self, learning_rate=0.001):
        input1 = tf.keras.Input(shape=(self.env.stateLen, ))
        x = tf.keras.layers.Dense(100, activation='relu')(input1)
        x = tf.keras.layers.Dropout(0.1)(x)
        output1 = tf.keras.layers.Dense(self.env.actionLen, activation='softmax')(x)
        actor = tf.keras.Model(inputs=input1, outputs=output1)
        actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.losses.categorical_crossentropy,
        )
        input2 = tf.keras.Input(shape=(self.env.stateLen, ))
        x = tf.keras.layers.Dense(100, activation='relu')(input2)
        x = tf.keras.layers.Dropout(0.1)(x)
        output2 = tf.keras.layers.Dense(1, activation='linear')(x)
        critic = tf.keras.Model(inputs=input2, outputs=output2)
        critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.losses.mse,
        )
        return actor, critic

class CartPole(CartPoleBase):
    modelNames = ['actor', 'critic']

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward = -1 if done else 0.01
        return next_state, reward, done

    def train(self, showProcess=False):
        agent = Agent(self, memorySize=1000)
        agent.actor.summary()
        agent.critic.summary()
        spendTime = 0
        episode = 0
        steps = [0] * 10
        config = self.load()
        if config:
            episode = config['episode']
            spendTime = config['spendTime']
            if config['actor']:
                agent.actor = config['actor']
            if config['critic']:
                agent.critic = config['critic']
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
                agent.save_exp(state, action, reward, next_state)
                step += 1
                paths.append(action)
                state = next_state
                if step % 5 == 0:
                    agent.learn()
                showProcess and self.render(0.016)
                if done:
                    break
            self.log(f'episode {episode}: step {step}, paths: {paths}')
            agent.learn()
            self.save([agent.actor, agent.critic], episode=episode, spendTime=spendTime)
            self.saveLog()
            steps[episode % 10] = step
            spendTime += time.time() - startTime
            if sum(steps) >= 1600:
                break
        return spendTime, episode

'''
与main_18主要不同:
使用记忆库学习,并且改为每五次或者游戏结束学习一次
统计:
    收敛次数        时间
1     135       00:06:38
2     399       00:14:56
3     312       00:17:17
4     298       00:15:08
5     
all   1144      00:53:59    
'''
if __name__ == '__main__':
    root = getDataFilePath(f'CartPole/CartPole_20/')
    if not os.path.exists(root):
        os.mkdir(root)
    startTime = time.time()
    totalEpisode = 0
    for i in range(5):
        cartPole = CartPole('CartPole-v0', os.path.join(root, f'CartPole_20_{i + 1}'))
        spendTime, episode = cartPole.train(showProcess=False)
        totalEpisode += episode
        print(f'train {i + 1}, spendTime {second2Str(int(spendTime))}, episode {episode}')
    print(f'testFinish spend {second2Str(int(time.time() - startTime))}, episode {totalEpisode}')
