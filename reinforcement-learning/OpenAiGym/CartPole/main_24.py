from lib import *
from OpenAiGym.CartPole.CartPoleBase import CartPoleBase
import tensorflow_probability as tfp

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
        self.actor, self.actor_opt, self.critic = self._createModel()
        self.memory = Memory()

    def chooseAction(self, state):
        prop = self.actor.predict(np.array([state]))[0]
        return np.random.choice(self.env.actionLen, p=prop)

    def save_exp(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self, gamma=0.95):
        sample = self.memory.sample()
        states, actions, rewards, next_states = vstack(sample)
        states, next_states = np.array(states), np.array(next_states)
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)
        v_predicts = [r + gamma * v[0] for r, v in zip(rewards, next_values)]
        td_errors = [vp - v[0] for vp, v in zip(v_predicts, values)]
        self.fitCritic(states, np.array(v_predicts))
        self.fitActor(states, np.array(td_errors), np.array(actions))
        self.memory.clear()

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
                agent.save_exp(state, action, reward, next_state)
                step += 1
                paths.append(action)
                state = next_state
                showProcess and self.render(0.016)
                if done:
                    break
            self.log(f'episode {episode}: step {step}, paths: {paths}')
            agent.learn()
            self.save([agent.actor, agent.critic], episode=episode, spendTime=spendTime)
            self.saveLog()
            steps[episode % 10] = step
            spendTime += time.time() - startTime
            if sum(steps) >= 1600:
                break
        return spendTime, episode

'''
与main_23主要不同:
学习率增加到0.001
统计:
    收敛次数        时间
1     111       00:04:37
2     56        00:02:11
3     58        00:02:20
4     46        00:01:45
5     52        00:02:18
all   323       00:13:14
'''
if __name__ == '__main__':
    root = getDataFilePath(f'CartPole/CartPole_24/')
    if not os.path.exists(root):
        os.mkdir(root)
    startTime = time.time()
    totalEpisode = 0
    for i in range(5):
        cartPole = CartPole('CartPole-v0', os.path.join(root, f'CartPole_24_{i + 1}'))
        spendTime, episode = cartPole.train(showProcess=False)
        totalEpisode += episode
        print(f'train {i + 1}, spendTime {second2Str(int(spendTime))}, episode {episode}')
    print(f'testFinish spend {second2Str(int(time.time() - startTime))}, episode {totalEpisode}')
