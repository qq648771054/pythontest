from lib import *
from OpenAiGym.CartPole.CartPoleBase import CartPoleBase

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.actor, self.critic = self._createModel()

    def chooseAction(self, state):
        prop = self.actor.predict(np.array([state]))[0]
        return np.random.choice(self.env.actionLen, p=prop)

    def learn(self, state, action, reward, next_state, gamma=0.95):
        state, next_state = np.array([state]), np.array([next_state])
        v_predict = reward + gamma * self.critic.predict(next_state)[0][0]
        td_error = v_predict - self.critic.predict(state)[0][0]
        action = [1 if i == action else 0 for i in range(self.env.actionLen)]
        self.critic.fit(state, np.array([v_predict]), epochs=1, verbose=0)
        self.actor.fit(state, np.array([action]), sample_weight=np.array([td_error]), epochs=1, verbose=0)

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
        agent = Agent(self)
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
                agent.learn(state, action, reward, next_state)
                step += 1
                paths.append(action)
                state = next_state
                showProcess and self.render(0.016)
                if done:
                    break
            self.log(f'episode {episode}: step {step}, paths: {paths}')
            self.save([agent.actor, agent.critic], episode=episode, spendTime=spendTime)
            self.saveLog()
            steps[episode % 10] = step
            spendTime += time.time() - startTime
            if sum(steps) >= 1600:
                break
        return spendTime, episode

'''
使用Actor-Critic
统计:
    收敛次数        时间
1     102       00:18:08
2     77        00:15:42
3     75        00:12:25
4     
5     
all   254       00:46:15
'''
if __name__ == '__main__':
    root = getDataFilePath(f'CartPole/CartPole_18/')
    if not os.path.exists(root):
        os.mkdir(root)
    startTime = time.time()
    totalEpisode = 0
    for i in range(5):
        cartPole = CartPole('CartPole-v0', os.path.join(root, f'CartPole_19_{i + 1}'))
        spendTime, episode = cartPole.train(showProcess=False)
        totalEpisode += episode
        print(f'train {i + 1}, spendTime {second2Str(int(spendTime))}, episode {episode}')
    print(f'testFinish spend {second2Str(int(time.time() - startTime))}, episode {totalEpisode}')
