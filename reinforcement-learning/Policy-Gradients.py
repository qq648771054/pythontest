from lib import *
import Thread
import envoriment

class Agent_Policy_Gradients(object):
    def __init__(self, env):
        self.env = env
        self.model = self._buildModel(self.env.stateShape, self.env.actionLen)
        self._ss = []
        self._as = []
        self._rs = []

    def save_exp(self, state, action, reward, next_state, done):
        self._ss.append(state)
        self._as.append(action)
        self._rs.append(reward)

    def learn(self, reward_decay=0.9):
        self.model.fit(np.array(self._ss), np.array(self._as),
                       sample_weight=self._discount_rewards(reward_decay),
                       verbose=0)
        self._ss = []
        self._as = []
        self._rs = []

    def choose_action(self, state):
        prop = self.model.predict(addAixs(state))[0]
        return np.random.choice(self.env.actionLen, p=prop)

    def _discount_rewards(self, reward_decay=0.9):
        """计算衰减reward的累加期望，并中心化和标准化处理"""
        prior = 0
        out = np.zeros_like(self._rs)
        for i in range(len(self._rs) - 1, -1, -1):
            prior = prior * reward_decay + self._rs[i]
            out[i] = prior
        return out / np.std(out - np.mean(out))

    def _buildModel(self, stateShape, actionLen, learning_rate=0.001):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=stateShape),
            # tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(actionLen, activation='softmax')
        ])
        model.compile(
            loss=tf.losses.mse,
            optimizer=tf.optimizers.RMSprop(learning_rate)
        )
        return model

class ThreadCartPole(Thread.ThreadBase):
    def run(self):
        env = envoriment.CartPole_v0(self.agentType)
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
    thread = ThreadCartPole(Agent_Policy_Gradients, showProcess=True)
    thread.start()
    while True:
        cmd = input()
        if cmd == '0':
            thread.showProcess = False
        else:
            thread.showProcess = True

