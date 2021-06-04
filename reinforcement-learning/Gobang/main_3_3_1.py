from lib import *
from Gobang.GobangBase import GobangBase

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
            self.weight.append(0)
        else:
            self.arr[self.iter] = m
            self.weight[self.iter] = 0
        self.newly.append(self.iter)
        self.iter = (self.iter + 1) % self.size

    def sample(self, batch):
        t = self.newly[:]
        self.newly = []
        if len(t) < batch:
            t.extend(np.random.choice(len(self.arr), size=batch - len(t), p=normallize(self.weight)))
        t.sort()
        idx, weight = [t[0]], [0]
        iter = 0
        for x in t:
            if idx[iter] != x:
                idx.append(x)
                weight.append(1)
                iter += 1
            else:
                weight[iter] += 1
        return idx, [self.arr[x] for x in idx], weight

    def updateErrors(self, idx, errors):
        for x, e in zip(idx, errors):
            self.weight[x] = abs(e) ** 0.5

    def clear(self):
        self.arr = []
        self.weight = []
        self.newly = []
        self.iter = 0

class Agent(object):
    def __init__(self, env, memorySize=300):
        self.env = env
        self.model = self._createModel()
        self.modelPre = copyModel(self.model)
        self.memory = Memory(memorySize)

    def chooseAction(self, board, greedy=0.95):
        actions = self.env.validActions(board)
        if random.random() < greedy:
            values = self.model.predict(self.addAxis(np.array([board])))[0]
            return self.findIndex(actions, values, lambda a, b: a > b)
        else:
            return np.random.choice(actions)

    def findIndex(self, actions, values, func=lambda a, b: a > b):
        ma = actions[0]
        for a in actions:
            if func(values[a], values[ma]):
                ma = a
        return ma

    def saveMemory(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def syncModel(self):
        self.modelPre = copyModel(self.model)

    def learn(self, batch=32, gamma=0.95):
        idxs, samples, weights = self.memory.sample(batch)
        states, actions, rewards, next_states = vstack(samples)
        a_states, a_next_states = self.addAxis(np.array(states)), self.addAxis(np.array(next_states))
        values = self.model.predict(a_states)
        next_values = self.model.predict(a_next_states)
        next_values_p = self.modelPre.predict(a_next_states)
        errors = []
        for v, nv, nvp, a, r, ns in zip(values, next_values, next_values_p, actions, rewards, next_states):
            actions = self.env.validActions(ns)
            if len(actions) > 0:
                idx = self.findIndex(actions, nv, lambda a, b: a > b)
                q = r + gamma * -nvp[idx]
            else:
                q = r
            errors.append(v[a] - q)
            v[a] = q
        self.model.fit(a_states, values, sample_weight=np.array(weights), epochs=1, verbose=0)
        self.memory.updateErrors(idxs, errors)

    def _createModel(self):
        input1 = tf.keras.Input(shape=self.env.stateSize)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        svalue = tf.keras.layers.Dense(1)(x)
        avalue = tf.keras.layers.Dense(self.env.actionSize)(x)
        mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(avalue)
        avalue = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([avalue, mean])
        output1 = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([svalue, avalue])
        model = tf.keras.Model(inputs=input1, outputs=output1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.losses.mse,
        )
        return model

    def addAxis(self, board):
        return board.reshape(board.shape + (1, ))

class Gobang(GobangBase):
    def step(self, action):
        next_state, player, winner = self.env.step(action)
        if winner is not None:
            if winner == player:
                reward = 1
            elif player == 1:
                reward = -0.1
            else:
                reward = 0.1
            return next_state, reward, player, True
        else:
            return next_state, 0, player, False

    def train(self, showProcess=False):
        agent = Agent(self, memorySize=300)
        agent.model.summary()
        spendTime = 0
        episode = 0
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
            path = []
            state = self.env.reset()
            showProcess and self.render(1)
            while True:
                action = agent.chooseAction(state)
                next_state, reward, player, done = self.step(action)
                agent.saveMemory(state, action, reward, next_state)
                step += 1
                path.append(action)
                state = next_state
                showProcess and self.render(0.5)
                if done:
                    break
            agent.learn(512 * 8)
            if episode % 10 == 0:
                agent.syncModel()
            self.log(f'episode {episode}: step {step}, path {path}, winner {player if step < self.env.actionSize else 0}')
            self.save(agent.model, episode=episode, spendTime=spendTime)
            self.saveLog()
            spendTime += time.time() - startTime
        # return spendTime, episode

'''
统计:
未见收敛
'''

if __name__ == '__main__':
    game = Gobang(3, 3, savePath=getDataFilePath('Gobang/Gobang_3_3/Gobang_3_3_1'))
    # game.play(1, Agent)
    game.train(showProcess=False)
