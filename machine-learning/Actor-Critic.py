from lib import *
import tkinter as tk
import threading
import gym


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus= tf.config.list_physical_devices('GPU')
if len(gpus) > 0: tf.config.experimental.set_memory_growth(gpus[0], True)

class Agent_Actor_Critic(object):
    def __init__(self, env):
        self.env = env
        self.actor = self._buildModelActor(self.env.stateShape, self.env.actionLen)
        self.critic = self._buildModelCritic(self.env.stateShape, self.env.actionLen)
        self._exp = None

    def save_exp(self, state, action, reward, next_state, done):
        self._exp = (state, action, reward, next_state, done)

    def learn(self):
        state, action, reward, next_state, done = self._exp
        state, next_state = addAixs(state), addAixs(next_state)
        q_predict = self.critic.predict(addAixs(state))[0][0]
        if done:
            td_error = reward - q_predict
        else:
            td_error = reward + self.critic.predict(next_state)[0][0] - q_predict
        self.critic.fit(state, np.array([td_error]), verbose=0)
        self.actor.fit(state, np.array([action]), sample_weight=np.array(td_error), verbose=0)

    def choose_action(self, state):
        prop = self.actor.predict(addAixs(state))[0]
        return np.random.choice(self.env.actionLen, p=prop)

    def _buildModelActor(self, stateShape, actionLen, learning_rate=0.001):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=stateShape),
            tf.keras.layers.Dense(actionLen, activation='softmax')
        ])
        model.compile(
            loss=tf.losses.mse,
            optimizer=tf.optimizers.RMSprop(learning_rate)
        )
        return model

    def _buildModelCritic(self, stateShape, actionLen, learning_rate=0.001):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=stateShape),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(
            loss=tf.losses.mse,
            optimizer=tf.optimizers.RMSprop(learning_rate)
        )
        return model

class EnvNN(object):
    def __init__(self, agentType):
        self.agent = agentType(self)

    @property
    def actionLen(self):
        raise NotImplementedError

    @property
    def stateShape(self):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

class EnvOpenAI(EnvNN):
    def __init__(self, env, agentType):
        self._env = env
        super(EnvOpenAI, self).__init__(agentType)

    @property
    def actionLen(self):
        return self._env.action_space.n

    @property
    def stateShape(self):
        return self._env.observation_space.shape

    def reset(self):
        return self._env.reset()

    def render(self):
        self._env.render()

    def step(self, action):
        return self._env.step(action)[: 3]

class CartPole_v0(EnvOpenAI):
    def step(self, action):
        next_state, reward, done = super(CartPole_v0, self).step(action)
        x, x_dot, theta, theta_dot = next_state
        env = self._env
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return next_state, reward, done

class EnvTk(EnvNN, tk.Tk):
    ACTION = []

    def __init__(self, agentType):
        tk.Tk.__init__(self)
        EnvNN.__init__(self, agentType)

    @property
    def actionLen(self):
        return len(self.ACTION)

    def render(self):
        self.update()

class ThreadBase(threading.Thread):
    def __init__(self, showProcess=True, savePath='', **kwargs):
        threading.Thread.__init__(self)
        self.showProcess = showProcess
        self.savePath = savePath
        self.args = kwargs

    def loadModel(self, agent):
        if self.savePath and os.path.exists(self.savePath):
            agent.model = tf.keras.models.load_model(self.savePath)

    def saveModel(self, agent):
        if self.savePath:
            agent.model.save(self.savePath)

    def render(self, env, sleepTime=None):
        if self.showProcess:
            env.render()
            if sleepTime:
                time.sleep(sleepTime)

class ThreadCartPole(ThreadBase):
    def run(self):
        env = CartPole_v0(gym.make('CartPole-v0'), Agent_Actor_Critic)
        agent = env.agent
        self.loadModel(agent)
        episode = 0
        while True:
            state = env.reset()
            self.render(env, 0.5)
            episode += 1
            step = 0
            while True:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.save_exp(state, action, reward, next_state, done)
                step += 1
                state = next_state
                self.render(env)
                if done:
                    break
            agent.learn()
            print('episode {}, steps {}'.format(episode, step))
            self.saveModel(agent)

if __name__ == '__main__':
    # thread = ThreadMaze(showProcess=False, savePath=getDataFilePath('dqn_maze_record.h5'))
    thread = ThreadCartPole(showProcess=True, savePath=getDataFilePath('dqn_cartPole_record.h5'))
    thread.start()
    while True:
        cmd = input()
        if cmd == '0':
            thread.showProcess = False
        else:
            thread.showProcess = True

