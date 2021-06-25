import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import threading
import gym
import datetime
import os
import tensorlayer as tl

episodes = 200000
gamma = 0.9
learning_rate = 1e-3
num_workers = 1

game = 'Breakout-ram-v0'
state_shape = (None, 128)
num_actions = 4

# game = 'SpaceInvaders-v4'
# state_shape = (None, 210, 160, 3)
# num_actions = 6

EPISODE = 0

def createActor():
    input = tf.keras.layers.Input(shape=state_shape)
    x = tf.keras.layers.Dense(512, activation='relu')(input)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output = tf.keras.layers.Dense(num_actions, activation='linear')(x)
    return tf.keras.Model(inputs=input, outputs=output)

def createCritic():
    input = tf.keras.layers.Input(shape=state_shape)
    x = tf.keras.layers.Dense(512, activation='relu')(input)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs=input, outputs=output)

# class CNNModel(tf.keras.models.Model):
#     def __init__(self, num_actions):
#         super(CNNModel, self).__init__()
#         # self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME')
#         # self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
#         #
#         # self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='SAME')
#         # self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
#         #
#         # self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='SAME')
#         # self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
#         #
#         # self.flatten = tf.keras.layers.Flatten()
#         self.linear1 = tf.keras.layers.Dense(units=512, activation='relu')
#         self.linear2 = tf.keras.layers.Dense(units=512, activation='relu')
#         self.actor_linear = tf.keras.layers.Dense(units=num_actions, activation='linear')
#         self.critic_linear = tf.keras.layers.Dense(units=1, activation='linear')
#
#
#     def call(self, inputs, training=None, mask=None):
#         # x = self.conv1(inputs)
#         # x = self.pool1(x)
#         # x = self.conv2(x)
#         # x = self.pool2(x)
#         # x = self.conv3(x)
#         # x = self.pool3(x)
#         # x = self.flatten(x)
#         x = self.linear1(inputs)
#         x = self.linear2(x)
#         return self.actor_linear(x), self.critic_linear(x)

def run_worker(global_actor, global_critic, global_actor_optimizer, global_critic_optimizer):
    global EPISODE
    global rewards_plt
    global losses_plt
    env = gym.make(game)
    actor, critic = createActor(), createCritic()

    while EPISODE < episodes:
        all_reward = 0
        states, values, act_probs, actions, rewards = [], [], [], [], []

        state = np.array([env.reset()], dtype=np.float) / 255.0

        actor.set_weights(global_actor.get_weights())
        critic.set_weights(global_critic.get_weights())
        steps = 0
        while True:
            states.append(state[0])
            act_prob, value = actor(state), critic(state)
            act_probs.append(act_prob[0])
            values.append(value[0])
            policy = tf.nn.softmax(act_prob)
            # if steps == 0:
            #     print(np.array(policy[0]))
            action = np.random.choice(num_actions, p=np.array(policy[0]))
            actions.append(action)
            state, reward, done, _ = env.step(action)
            state = state / 255.0
            steps += 1
            rewards.append(reward)

            all_reward += reward
            state = np.array([state], dtype=np.float)
            if done: break

        rewards_plt.append(all_reward)

        _values = []
        for i in range(len(values) - 1):
            _values.append([rewards[i] + gamma * values[i + 1][0]])
        _values.append([rewards[-1]])

        # act_prob, value = actor(np.array(states, dtype=np.float)), critic(np.array(states, dtype=np.float))
        # policy = tf.nn.softmax(act_prob)
        with tf.GradientTape() as tape:
            advantages = np.array(_values) - np.array(values)
            advantages = np.reshape(advantages, newshape=(-1))
            actions_onehot = np.eye(num_actions)[actions]
            act_prob = actor(np.array(states, dtype=np.float))
            losses_actor = advantages * tf.nn.softmax_cross_entropy_with_logits(labels=actions_onehot, logits=act_prob)
            grad_actor = tape.gradient(tf.reduce_mean(losses_actor), actor.trainable_variables)
            global_actor_optimizer.apply_gradients(zip(grad_actor, actor.trainable_variables))

        with tf.GradientTape() as tape:
            value = critic(np.array(states, dtype=np.float))
            losses_critic = 0.5 * tf.reshape((value - _values) ** 2, shape=(-1))
            grad_critic = tape.gradient(tf.reduce_mean(losses_critic), global_critic.trainable_variables)
            global_critic_optimizer.apply_gradients(zip(grad_critic, global_critic.trainable_variables))

        print('{}:episode {}; step {}; rewards {}; losses {};'.format(
            str(datetime.datetime.now()),
            EPISODE + 1,
            steps,
            all_reward,
            [tf.reduce_mean(losses_actor), tf.reduce_mean(losses_critic)],
        ))
        EPISODE += 1

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    global_actor, global_critic = createActor(), createCritic()
    global_actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    global_critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    rewards_plt = []

    threads = []
    for _ in range(num_workers):
        p = threading.Thread(target=run_worker, args=[global_actor, global_critic,
                                                      global_actor_optimizer, global_critic_optimizer])
        p.start()
        threads.append(p)
    for p in threads: p.join()

    plt.plot(rewards_plt)
    plt.show()
