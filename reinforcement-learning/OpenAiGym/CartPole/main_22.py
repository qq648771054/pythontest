from lib import *
from OpenAiGym.CartPole.CartPoleBase import CartPoleBase
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
        self.actor, self.critic = self._createModel()
        self.memory = Memory()

    def chooseAction(self, state):
        prop = self.actor.predict(np.array([state]))[0]
        return np.random.choice(self.env.actionLen, p=prop)

    def save_exp(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def pop_exp(self):
        m = self.memory.sample()
        self.memory.clear()
        return m

    def learn(self, sample, gamma=0.95):
        states, actions, rewards, next_states, dones = vstack(sample)
        states, next_states = np.array(states), np.array(next_states)
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)
        v_predicts = [r if d else r + gamma * v[0] for r, v, d in zip(rewards, next_values, dones)]
        td_errors = [vp - v[0] for vp, v in zip(v_predicts, values)]
        actions = [[1 if i == a else 0 for i in range(self.env.actionLen)] for a in actions]
        self.critic.fit(states, np.array(v_predicts), epochs=1, verbose=0)
        self.actor.fit(states, np.array(actions), sample_weight=np.array(td_errors), epochs=1, verbose=0)

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

class Worker(threading.Thread):
    def __init__(self, parentAgent, game, env, idx):
        threading.Thread.__init__(self)
        self.parentAgent = parentAgent
        self.env = env
        self.game = game
        self.agent = Agent(env)
        copyModel(self.agent.actor, self.parentAgent.actor)
        copyModel(self.agent.critic, self.parentAgent.critic)
        self.idx = idx

    def run(self):
        while True:
            step = 0
            paths = []
            state = self.game.reset()
            while True:
                action = self.agent.chooseAction(state)
                next_state, reward, done = self.env.warpper(*self.game.step(action))
                self.agent.save_exp(state, action, reward, next_state, done)
                step += 1
                paths.append(action)
                state = next_state
                if done:
                    break
            self.parentAgent.learn(self.agent.pop_exp())
            copyModel(self.agent.actor, self.parentAgent.actor)
            copyModel(self.agent.critic, self.parentAgent.critic)
            self.env.episode += 1
            self.env.log(f'agent {self.idx}, episode {self.env.episode}: step {step}, paths: {paths}')
            if self.env.episode % 30 == 0:
                self.env.saveModel()

class CartPole(CartPoleBase):
    modelNames = ['actor', 'critic']

    def warpper(self, next_state, reward, done, info):
        return next_state, reward, done

    def train(self, childCnt=8):
        self.agent = Agent(self)
        self.agent.actor.summary()
        self.agent.critic.summary()
        config = self.load()
        self.episode = 0
        self.spendTime = 0
        self.savedTime = time.time()
        if config:
            self.episode = config['episode']
            self.spendTime = config['spendTime']
            if config['actor']:
                self.agent.actor = config['actor']
            if config['critic']:
                self.agent.critic = config['critic']
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
            if config['actor']:
                self.agent.actor = config['actor']
            if config['critic']:
                self.agent.critic = config['critic']
        episode = 0
        while True:
            step = 0
            paths = []
            state = self.env.reset()
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
        self.save([self.agent.actor, self.agent.critic], episode=self.episode, spendTime=self.spendTime)
        self.saveLog()

'''
与main_21主要不同:
改变reward算法,变成得分
'''
if __name__ == '__main__':
    root = getDataFilePath(f'CartPole/CartPole_22/')
    if not os.path.exists(root):
        os.mkdir(root)
    cartPole = CartPole('CartPole-v0', os.path.join(root, f'CartPole_22_1'))
    # cartPole.train()
    cartPole.display()
