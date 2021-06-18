from lib import *
from OpenAiGym.CartPole.CartPoleBase import CartPoleBase
import tensorflow_probability as tfp
import multiprocessing
import gym
import threading

class Agent(object):
    GAMMA = 0.95

    def __init__(self, env):
        self.env = env
        self.actor, self.actor_opt, self.critic = self._createModel()

    def chooseAction(self, state):
        prop = self.actor.predict(np.array([state]))[0]
        return np.random.choice(self.env.actionLen, p=prop)

    def predict_v(self, state):
        return self.critic.predict(np.array([state]))[0][0]

    def learn(self, sample):
        states, actions, v_predicts = vstack(sample)
        states = np.array(states)
        values = self.critic.predict(states)
        td_errors = [vp - v[0] for vp, v in zip(v_predicts, values)]
        self.fitCritic(states, np.array(v_predicts))
        self.fitActor(states, np.array(td_errors), np.array(actions))

    def fitCritic(self, states, v_predicts):
        self.critic.fit(states, v_predicts, epochs=10, verbose=0)

    def fitActor(self, states, td_errors, actions):
        oldpi = self.actor(states)
        dist = tfp.distributions.Categorical(probs=oldpi)
        oldpi = dist.prob(actions)
        for i in range(10):
            with tf.GradientTape() as tape:
                newpi = self.actor(states)
                dist = tfp.distributions.Categorical(probs=newpi)
                newpi = dist.prob(actions)
                ratio = newpi / (oldpi + 1e-8)
                loss = -tf.reduce_mean(
                    tf.minimum(ratio * td_errors,
                               tf.clip_by_value(ratio, 0.8, 1.2) * td_errors)
                )
            a_gard = tape.gradient(loss, self.actor.trainable_weights)
            self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

    def _createModel(self):
        input1 = tf.keras.Input(shape=(self.env.stateLen, ))
        x = tf.keras.layers.Dense(100, activation='relu')(input1)
        x = tf.keras.layers.Dropout(0.1)(x)
        output1 = tf.keras.layers.Dense(self.env.actionLen, activation='softmax')(x)
        actor = tf.keras.Model(inputs=input1, outputs=output1)
        actor.compile()
        actor_opt = tf.keras.optimizers.Adam(0.001)
        input2 = tf.keras.Input(shape=(self.env.stateLen, ))
        x = tf.keras.layers.Dense(100, activation='relu')(input2)
        x = tf.keras.layers.Dropout(0.1)(x)
        output2 = tf.keras.layers.Dense(1, activation='linear')(x)
        critic = tf.keras.Model(inputs=input2, outputs=output2)
        critic.compile(
            optimizer=tf.keras.optimizers.Adam(0.002),
            loss=tf.losses.mse,
        )
        return actor, actor_opt, critic

class Worker(threading.Thread):
    def __init__(self, parentAgent, game, env, workEvent, updateEvent, idx):
        threading.Thread.__init__(self)
        self.parentAgent = parentAgent
        self.env = env
        self.game = game
        self.workEvent = workEvent
        self.updateEvent = updateEvent
        self.idx = idx
        self.exp = []

    def run(self):
        exp = []
        while True:
            step = 0
            paths = []
            state = self.game.reset()
            while True:
                self.workEvent.wait()
                action = self.parentAgent.chooseAction(state)
                next_state, reward, done = self.env.warpper(*self.game.step(action))
                exp.append((state, action, reward))
                step += 1
                self.env.totalSteps += 1
                paths.append(action)
                state = next_state
                self.env.counter += 1
                if done or self.env.counter >= self.env.BATCH:
                    v_predict = []
                    v = 0 if done else self.parentAgent.predict_v(next_state)
                    for s, a, r in reversed(exp):
                        v = r + self.parentAgent.GAMMA * v
                        v_predict.append(v)
                    self.env.memory.extend([(e[0], e[1], v) for e, v in zip(exp, reversed(v_predict))])
                    exp = []
                    if self.env.counter >= self.env.BATCH:
                        self.workEvent.clear()
                        self.updateEvent.set()
                    if done:
                        break
            self.env.episode += 1
            self.env.log(f'agent {self.idx}, episode {self.env.episode} step {step}, totalSteps {self.env.totalSteps}, paths {paths}')

class Learner(threading.Thread):
    def __init__(self, env, workEvents, updateEvent, agent):
        threading.Thread.__init__(self)
        self.env = env
        self.workEvents = workEvents
        self.updateEvent = updateEvent
        self.agent = agent

    def run(self):
        learnCount = 0
        while True:
            self.updateEvent.wait()
            self.updateEvent.clear()
            setCount = 0
            for e in self.workEvents:
                setCount += int(e.is_set())
            if setCount != 0:
                continue
            self.agent.learn(self.env.memory)
            self.env.counter = 0
            self.env.memory = []
            for e in self.workEvents:
                e.set()
            learnCount += 1
            if learnCount % 50 == 0:
                self.env.saveModel()

class CartPole(CartPoleBase):
    modelNames = ['actor', 'critic']
    BATCH = 64

    def warpper(self, next_state, reward, done, info):
        reward = -1 if done else 0.01
        return next_state, reward, done

    def train(self, childCnt=multiprocessing.cpu_count()):
        self.agent = Agent(self)
        self.agent.actor.summary()
        self.agent.critic.summary()
        self.spendTime = 0
        self.episode = 0
        self.savedTime = time.time()
        config = self.load()
        if config:
            self.episode = config['episode']
            self.spendTime = config['spendTime']
            if config['actor']:
                self.agent.actor = config['actor']
            if config['critic']:
                self.agent.critic = config['critic']
        self.totalSteps = 0
        self.counter = 0
        self.memory = []
        workers = []
        envs = []
        workEvents = []
        updateEvent = threading.Event()
        for i in range(childCnt):
            env = gym.make(self.gameName)
            event = threading.Event()
            event.set()
            worker = Worker(self.agent, env, self, event, updateEvent, i)
            workers.append(worker)
            envs.append(env)
            workEvents.append(event)
        learner = Learner(self, workEvents, updateEvent, self.agent)
        learner.start()
        for worker in workers:
            worker.start()
        threads = workers[:] + [learner]
        coord = tf.train.Coordinator()
        coord.join(threads)

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
与main_24主要不同:
使用dppo
'''
if __name__ == '__main__':
    root = getDataFilePath(f'CartPole/CartPole_25/')
    if not os.path.exists(root):
        os.mkdir(root)
    cartPole = CartPole('CartPole-v0', os.path.join(root, f'CartPole_25_1'))
    cartPole.train()

