from lib import *
from OpenAiGym.BreakOut.BreakOutBase import BreakOutBase
import threading

# 降低tensorflow警告等级
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 配置GPU内存
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.actor, self.actor_opt, self.critic = self._createModel()

    def chooseAction(self, state):
        # prop = np.array(tf.nn.softmax(self.actor(np.array([state])))[0])
        # return np.random.choice(self.env.actionLen, p=prop), prop
        prop = self.actor.predict(np.array([state]))[0]
        return np.random.choice(self.env.actionLen, p=prop), prop

    def learn(self, samples, gamma=0.95):
        batch = 1000
        for i in range(0, len(samples), batch):
            sample = samples[i: i + batch]
            states, actions, rewards, next_states, dones = vstack(sample)
            states, next_states = np.array(states), np.array(next_states)
            values = self.critic.predict(states)
            next_values = self.critic.predict(next_states)
            v_predicts = [[r if d else r + gamma * v[0]] for r, v, d in zip(rewards, next_values, dones)]
            self.critic.fit(states, np.array(v_predicts), epochs=1, verbose=0)
            with tf.GradientTape() as tape:
                td_errors = [vp[0] - v[0] for vp, v in zip(v_predicts, values)]
                # actions = [[1 if i == a else 0 for i in range(self.env.actionLen)] for a in actions]
                props = self.actor(states)
                props = [max(p[a], 1e-8) for a, p in zip(actions, props)]
                losses = td_errors * tf.math.log(props)
                grad = tape.gradient(-tf.reduce_mean(losses), self.actor.trainable_weights)
                self.actor_opt.apply_gradients(zip(grad, self.actor.trainable_weights))
            # self.actor.fit(states, np.array(actions), epochs=1, verbose=0)

    def _createModel(self):
        input1 = tf.keras.Input(shape=self.env.stateSize)
        x = tf.keras.layers.Dense(512, activation='relu')(input1)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        output1 = tf.keras.layers.Dense(self.env.actionLen, activation='softmax')(x)
        actor = tf.keras.Model(inputs=input1, outputs=output1)
        actor.compile()
        actor_opt = tf.keras.optimizers.Adam(0.0001)
        input2 = tf.keras.Input(shape=self.env.stateSize)
        x = tf.keras.layers.Dense(512, activation='relu')(input2)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        output2 = tf.keras.layers.Dense(1, activation='linear')(x)
        critic = tf.keras.Model(inputs=input2, outputs=output2)
        critic.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.mse
        )
        return actor, actor_opt, critic

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
            rewards = 0
            state = self.game.reset() / 255.0
            memory = []
            actions = []
            while True:
                action, prop = self.agent.chooseAction(state)
                if step == 0:
                    print(prop)
                next_state, reward, done = self.env.step(self.game, action)
                next_state = next_state / 255.0
                memory.append((state, action, reward, next_state, done))
                rewards += reward
                step += 1
                self.env.totalStep += 1
                actions.append(action)
                state = next_state
                if done:
                    break
            self.parentAgent.learn(memory)
            copyModel(self.agent.actor, self.parentAgent.actor)
            copyModel(self.agent.critic, self.parentAgent.critic)
            self.env.episode += 1
            self.env.log(f'agent: {self.idx}, episode: {self.env.episode}, step: {step}'
                         f', totalStep: {self.env.totalStep}, rewards {round(rewards)}, actions {actions}')
            if self.env.episode % 10 == 0:
                self.env.saveModel()

class Game(BreakOutBase):
    modelNames = ['actor', 'critic']

    def step(self, game, action):
        next_state, reward, done, info = game.step(action)
        return next_state, reward, done

    def train(self, childCnt=1):
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

        workers = []
        envs = []
        for i in range(childCnt):
            env = self.makeEnv(self.gameName)
            worker = Worker(self.agent, env, self, i)
            workers.append(worker)
            envs.append(env)

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

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
            state = self.env.reset() / 255.0
            while True:
                self.render()
                action = self.agent.chooseAction(state)
                next_state, reward, done = self.step(self.env, action)
                next_state = next_state / 255.0
                rewards += reward
                step += 1
                state = next_state
                if done:
                    break
            print(f'step: {step}, rewards {round(rewards)}')

    def saveModel(self):
        self.spendTime += time.time() - self.savedTime
        self.savedTime = time.time()
        self.save([self.agent.actor, self.agent.critic], episode=self.episode,
                  spendTime=self.spendTime, totalStep=self.totalStep)
        self.saveLog()

'''
使用a3c
'''
if __name__ == '__main__':
    root = getDataFilePath(f'BreakOut/BreakOut_1/')
    if not os.path.exists(root):
        os.mkdir(root)
    game = Game('Breakout-ram-v0', os.path.join(root, f'BreakOut_1_3'))
    game.train()
    # game.display()
