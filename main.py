from collections import deque

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import copy
import gym

class ActorCritic:
    def __init__(self, actor_model, critic_model):
        self.actor_model = actor_model
        self.critic_model = critic_model
        actor_optimizer = tf.keras.optimizers.Adam(1e-3)
        critic_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.actor_model.compile(
            loss='categorical_crossentropy',
            optimizer=actor_optimizer
        )
        self.critic_model.compile(
            loss='mse',
            optimizer=critic_optimizer
        )

        self.replay_buffer = deque(maxlen=int(1e6))
        self.simple_size = 8192
        self.gamma = 0.997


    def save_memory(self, state, action_onehot, reward, next_state, done):

        self.replay_buffer.append(
            (
                np.array([state], dtype=np.float64),
                action_onehot, reward,
                np.array([next_state], dtype=np.float64),
                done
            )
        )

    def choice_action(self, state):
        prob = np.array(self.actor_model(np.array([state], dtype=np.float64))[0])
        return np.random.choice(len(prob), p=prob)

    def update_weights(self):
        batch_size = min(self.simple_size, len(self.replay_buffer))
        training_data = random.sample(self.replay_buffer, batch_size)

        batch_states = []
        batch_values = []
        batch_policy = []
        for data in training_data:
            state, action_onehot, reward, next_state, done = data
            batch_states.append(state[0])
            value_target = reward if done else reward + self.gamma * np.array(self.critic_model(next_state))[0][0]

            td_error = value_target - np.array(self.critic_model(state))[0][0]
            batch_values.append([value_target])
            policy_target = td_error * np.array(action_onehot)
            batch_policy.append(policy_target)

        batch_states = np.array(batch_states)
        batch_values = np.array(batch_values)
        batch_policy = np.array(batch_policy)

        self.critic_model.fit(batch_states, batch_values, epochs=1, verbose=0)
        self.actor_model.fit(batch_states, batch_policy, epochs=1, verbose=0)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    actor_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=2, activation='softmax'),
    ])
    actor_model.build(input_shape=(None, 4))
    critic_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear'),
    ])
    critic_model.build(input_shape=(None, 4))
    agent = ActorCritic(actor_model, critic_model)

    episodes = 200
    all_rewards = []

    for e in range(episodes):

        state = env.reset()

        rewards = 0
        path = []
        while True:
            # env.render()
            action = agent.choice_action(state)
            next_state, reward, done, _ = env.step(action)
            action_onehot = [1 if i == action else 0 for i in range(2)]
            agent.save_memory(state, action_onehot, reward, next_state, done)

            state = copy.deepcopy(next_state)

            rewards += reward
            path.append(action)
            if done:
                print("e: {}/{}, rewards: {} path {}".format(e + 1, episodes, rewards, path))
                all_rewards.append(rewards)
                break

        agent.update_weights()

    plt.plot(all_rewards)
    plt.show()
