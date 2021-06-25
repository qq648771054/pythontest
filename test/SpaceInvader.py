from tensorflow.keras import Sequential, layers
from collections import deque

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import random
import copy
import gym
import os

from Lib import *

# 降低tensorflow警告等级
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 配置GPU内存
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

class DQNAgent:
    def __init__(self, env):
        self._env = env
        self.memory = deque(maxlen=2048)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.9

        self.simple_size = 64

        self.model = self._Build_Deep_Q_Network()

    def _Build_Deep_Q_Network(self):
        model = Sequential()

        model.add(layers.Conv2D(filters=16, kernel_size=(8, 8), strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

        model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), strides=2, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

        model.add(layers.Flatten())
        model.add(layers.Dense(units=256, activation='relu'))
        model.add(layers.Dense(units=6))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        model.build(input_shape=(None, 210, 160, 3))
        model.summary()
        return model

    def save_memory(self, state, action, reward, next_state, done):

        _state = state/255.0
        _next_state = next_state/255.0

        self.memory.append((_state, action, reward, _next_state, done))


    def train(self):
        batchs = min(self.simple_size, len(self.memory))
        training_data = random.sample(self.memory, batchs)
        states, actions, rewards, next_states, dones = vstack(training_data)
        states, next_states = np.array(states), np.array(next_states)
        values = self.model.predict(states)
        next_values = self.model.predict(next_states)
        for r, a, v, nv, d in zip(rewards, actions, values, next_values, dones):
            p = r if d else r + self.gamma * nv.max()
            v[a] = p
        self.model.fit(states, values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def choice_action(self, state):
        _state = np.array([state], dtype=np.float64)/255.0
        if np.random.rand() > self.epsilon:
            return self._env.action_space.sample()
        else:
            action = self.model.predict(_state)
            return np.argmax(action[0])

if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    agent = DQNAgent(env)

    if os.path.exists('./save_weights.h5'):
        agent.model.load_weights('./save_weights.h5')

    plt_rewards = []
    episodes = 10000
    for e in range(episodes):
        state = env.reset()
        all_rewards = 0
        actions = []
        for time_t in range(5000):

            env.render()

            action = agent.choice_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = -100 if done else reward
            agent.save_memory(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            all_rewards += reward
            actions.append(action)
            if done:
                print("{}:episode: {}/{}, all rewards: {}, steps: {}, epsilon: {}, action: {}"
                      .format(datetime.datetime.now, e + 1, episodes, all_rewards, time_t, agent.epsilon, actions))
                break
        plt_rewards.append(all_rewards)
        agent.train()

        if (e+1) % 100 == 0:
            model = './save_weights.h5'
            print('保存模型:'+model)
            agent.model.save_weights(model)

    plt.plot(plt_rewards)
    plt.show()