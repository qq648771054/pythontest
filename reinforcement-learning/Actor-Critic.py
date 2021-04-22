from lib import *
import Thread
import envoriment

class Agent_Actor_Critic(object):
    def __init__(self, env):
        self.env = env
        self.actor = self._buildModelActor(self.env.stateShape, self.env.actionLen)
        self.critic = self._buildModelCritic(self.env.stateShape, self.env.actionLen)
        self._exp = None

    def save_exp(self, state, action, reward, next_state, done):
        self._exp = (state, action, reward, next_state, done)

    def learn(self):
        import copy
        state, action, reward, next_state, done = self._exp
        state, next_state = addAixs(state), addAixs(next_state)
        q_predict = self.critic.predict(state)[0][0]
        if done:
            td_error = reward - q_predict
        else:
            td_error = reward + self.critic.predict(next_state)[0][0] - q_predict
        self.critic.fit(state, np.array([td_error]), verbose=0)
        self.actor.fit(copy.deepcopy(state), np.array([action]), sample_weight=np.array([td_error]), verbose=0)

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
            tf.keras.layers.Dense(10, activation='relu', input_shape=stateShape),
            tf.keras.layers.Dense(1, activation='linear')
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
                agent.learn()
                step += 1
                state = next_state
                self.render(env)
                if done:
                    break

            print('episode {}, steps {}'.format(episode, step))
            self.saveModel(agent)

if __name__ == '__main__':
    thread = ThreadCartPole(Agent_Actor_Critic, showProcess=True)
    thread.start()
    while True:
        cmd = input()
        if cmd == '0':
            thread.showProcess = False
        else:
            thread.showProcess = True

