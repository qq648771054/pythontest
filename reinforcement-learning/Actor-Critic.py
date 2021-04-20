from lib import *
import thread

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
            tf.keras.layers.Dense(128, activation='relu', input_shape=stateShape),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(
            loss=tf.losses.mse,
            optimizer=tf.optimizers.RMSprop(learning_rate)
        )
        return model

if __name__ == '__main__':
    # thread = ThreadMaze(showProcess=False, savePath=getDataFilePath('dqn_maze_record.h5'))
    thread = thread.ThreadCartPole(Agent_Actor_Critic, showProcess=True)
    thread.start()
    while True:
        cmd = input()
        if cmd == '0':
            thread.showProcess = False
        else:
            thread.showProcess = True

