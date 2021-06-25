from lib import *
from OpenAiGym.SpaceInvaders.SpaceInvadersBase import SpaceInvadersBase
import tensorflow_probability as tfp
import multiprocessing
import gym
import threading

# 降低tensorflow警告等级
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 配置GPU内存
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

class Agent(object):
    GAMMA = 0.95
    LEARNTIME = 20

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
        print('predict', v_predicts.tolist())
        self.critic.fit(states, v_predicts, epochs=self.LEARNTIME, verbose=0)

    def fitActor(self, states, td_errors, actions):
        oldp = self.actor(states)
        dist = tfp.distributions.Categorical(probs=oldp)
        oldpi = dist.prob(actions)
        for i in range(self.LEARNTIME):
            with tf.GradientTape() as tape:
                newp = self.actor(states)
                dist = tfp.distributions.Categorical(probs=newp)
                newpi = dist.prob(actions)
                ratio = newpi / (oldpi + 1e-8)
                loss = -tf.reduce_mean(
                    tf.minimum(ratio * td_errors,
                               tf.clip_by_value(ratio, 0.8, 1.2) * td_errors)
                )
            a_gard = tape.gradient(loss, self.actor.trainable_weights)
            self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))
            s = self.actor.predict(states)
            for i in s:
                for j in i:
                    if math.isnan(j):
                        print(1)

    def _createModel(self):
        input1 = tf.keras.Input(shape=self.env.stateSize)
        x = tf.keras.layers.Conv2D(32, (5, 5), 2, activation='relu', padding='same')(input1)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (5, 5), 2, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (4, 4), 2, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), 1, activation='relu', padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x1 = tf.keras.layers.Dense(512, activation='relu')(x)
        output1 = tf.keras.layers.Dense(self.env.actionLen, activation='softmax')(x1)
        actor = tf.keras.Model(inputs=input1, outputs=output1)
        actor.compile()
        actor_opt = tf.keras.optimizers.Adam(0.0001)
        x2 = tf.keras.layers.Dense(512, activation='relu')(x)
        output2 = tf.keras.layers.Dense(1, activation='linear')(x2)
        critic = tf.keras.Model(inputs=input1, outputs=output2)
        critic.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss=tf.keras.losses.mse
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
            rewards = 0
            actions = []
            state = rgb2Gray(self.game.reset()) / 255.0
            while True:
                self.workEvent.wait()
                action = self.parentAgent.chooseAction(state)
                next_state, reward, done = self.env.step(self.game, action)
                next_state = rgb2Gray(next_state) / 255.0
                exp.append((state, action, reward))
                rewards += reward
                step += 1
                actions.append(action)
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
            self.env.totalStep += step
            self.env.episode += 1
            self.env.log(f'agent: {self.idx}, episode: {self.env.episode}, step: {step}'
                         f', totalStep: {self.env.totalStep}, rewards {round(rewards + 1000)}, actions {actions}')

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
            if learnCount % 100 == 0:
                self.env.saveModel()

class Game(SpaceInvadersBase):
    modelNames = ['actor', 'critic']
    stateSize = (210, 160, 1)
    BATCH = 64

    def step(self, game, action):
        next_state, reward, done, info = game.step(action)
        if info['ale.lives'] == 0:
            reward = -1000
        reward -= 0.1
        return next_state, reward, done

    def train(self, childCnt=multiprocessing.cpu_count()):
        self.agent = Agent(self)
        self.agent.actor.summary()
        self.agent.critic.summary()
        self.spendTime = 0
        self.episode = 0
        self.totalStep = 0
        self.savedTime = time.time()
        config = self.load()
        if config:
            self.episode = config['episode']
            self.spendTime = config['spendTime']
            self.totalStep = config['totalStep']
            if config['actor']:
                self.agent.actor = config['actor']
            if config['critic']:
                self.agent.critic = config['critic']
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
        self.agent.actor.summary()
        self.agent.critic.summary()
        config = self.load()
        self.episode = 0
        self.spendTime = 0
        self.savedTime = time.time()
        if config:
            self.episode = config['episode']
            self.spendTime = config['spendTime']
            self.totalStep = config['totalStep']
            if config['actor']:
                self.agent.actor = config['actor']
            if config['critic']:
                self.agent.critic = config['critic']
        while True:
            step = 0
            rewards = 0
            state = rgb2Gray(self.env.reset()) / 255.0
            while True:
                self.render()
                action = self.agent.chooseAction(state)
                next_state, reward, done = self.step(self.env, action)
                next_state = rgb2Gray(next_state) / 255.0
                rewards += reward
                step += 1
                state = next_state
                if done:
                    break
            print(f'step: {step}, rewards {round(rewards + 1000)}')

    def saveModel(self):
        self.spendTime += time.time() - self.savedTime
        self.savedTime = time.time()
        self.save([self.agent.actor, self.agent.critic], episode=self.episode,
                  spendTime=self.spendTime, totalStep=self.totalStep)
        self.saveLog()

'''
使用dppo算法
'''
if __name__ == '__main__':
    root = getDataFilePath(f'SpaceInvaders/SpaceInvaders_3/')
    if not os.path.exists(root):
        os.mkdir(root)
    game = Game('SpaceInvaders-v0', os.path.join(root, f'SpaceInvaders_3_3'))
    # game.train()
    game.display()
