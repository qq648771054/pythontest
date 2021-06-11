# from collections import deque
#
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np
# import random
# import copy
# import gym
#
# class ActorCritic:
#     def __init__(self, actor_model, critic_model):
#         self.actor_model = actor_model
#         self.critic_model = critic_model
#         actor_optimizer = tf.keras.optimizers.Adam(1e-3)
#         critic_optimizer = tf.keras.optimizers.Adam(1e-3)
#         self.actor_model.compile(
#             loss='categorical_crossentropy',
#             optimizer=actor_optimizer
#         )
#         self.critic_model.compile(
#             loss='mse',
#             optimizer=critic_optimizer
#         )
#
#         self.replay_buffer = deque(maxlen=int(1e6))
#         self.simple_size = 8192
#         self.gamma = 0.997
#
#
#     def save_memory(self, state, action_onehot, reward, next_state, done):
#
#         self.replay_buffer.append(
#             (
#                 np.array([state], dtype=np.float64),
#                 action_onehot, reward,
#                 np.array([next_state], dtype=np.float64),
#                 done
#             )
#         )
#
#     def choice_action(self, state):
#         prob = np.array(self.actor_model(np.array([state], dtype=np.float64))[0])
#         return np.random.choice(len(prob), p=prob)
#
#     def update_weights(self):
#         batch_size = min(self.simple_size, len(self.replay_buffer))
#         training_data = random.sample(self.replay_buffer, batch_size)
#
#         batch_states = []
#         batch_values = []
#         batch_policy = []
#         for data in training_data:
#             state, action_onehot, reward, next_state, done = data
#             batch_states.append(state[0])
#             value_target = reward if done else reward + self.gamma * np.array(self.critic_model(next_state))[0][0]
#
#             td_error = value_target - np.array(self.critic_model(state))[0][0]
#             batch_values.append([value_target])
#             policy_target = td_error * np.array(action_onehot)
#             batch_policy.append(policy_target)
#
#         batch_states = np.array(batch_states)
#         batch_values = np.array(batch_values)
#         batch_policy = np.array(batch_policy)
#
#         self.critic_model.fit(batch_states, batch_values, epochs=1, verbose=0)
#         self.actor_model.fit(batch_states, batch_policy, epochs=1, verbose=0)
#
# if __name__ == '__main__':
#     env = gym.make('CartPole-v1')
#     actor_model = tf.keras.Sequential([
#         tf.keras.layers.Dense(units=128, activation='relu'),
#         tf.keras.layers.Dense(units=2, activation='softmax'),
#     ])
#     actor_model.build(input_shape=(None, 4))
#     critic_model = tf.keras.Sequential([
#         tf.keras.layers.Dense(units=128, activation='relu'),
#         tf.keras.layers.Dense(units=1, activation='linear'),
#     ])
#     critic_model.build(input_shape=(None, 4))
#     agent = ActorCritic(actor_model, critic_model)
#
#     episodes = 200
#     all_rewards = []
#
#     for e in range(episodes):
#
#         state = env.reset()
#
#         rewards = 0
#
#         while True:
#             env.render()
#             action = agent.choice_action(state)
#             next_state, reward, done, _ = env.step(action)
#             action_onehot = [1 if i == action else 0 for i in range(2)]
#             agent.save_memory(state, action_onehot, reward, next_state, done)
#
#             state = copy.deepcopy(next_state)
#
#             rewards += reward
#
#             if done:
#                 print("e: {}/{}, rewards: {}".format(e + 1, episodes, rewards))
#                 all_rewards.append(rewards)
#                 break
#
#         agent.update_weights()
#
#     plt.plot(all_rewards)
#     plt.show()
#
#
#
# #######################################################################
# # Copyright (C)                                                       #
# # 2016 - 2019 Pinard Liu(liujianping-ok@163.com)                      #
# # https://www.cnblogs.com/pinard                                      #
# # Permission given to modify the code as long as you keep this        #
# # declaration at the top                                              #
# #######################################################################
# ## https://www.cnblogs.com/pinard/p/10272023.html ##
# ## 强化学习(十四) Actor-Critic ##
#
# import gym
# import tensorflow as tf
# import numpy as np
# import random
# from collections import deque
#
# # Hyper Parameters
# GAMMA = 0.95  # discount factor
# LEARNING_RATE = 0.01
#
#
# class Actor():
#     def __init__(self, env, sess):
#         # init some parameters
#         self.time_step = 0
#         self.state_dim = env.observation_space.shape[0]
#         self.action_dim = env.action_space.n
#         self.create_softmax_network()
#
#         # Init session
#         self.session = sess
#         self.session.run(tf.global_variables_initializer())
#
#     def create_softmax_network(self):
#         # network weights
#         W1 = self.weight_variable([self.state_dim, 20])
#         b1 = self.bias_variable([20])
#         W2 = self.weight_variable([20, self.action_dim])
#         b2 = self.bias_variable([self.action_dim])
#         # input layer
#         self.state_input = tf.placeholder("float", [None, self.state_dim])
#         self.tf_acts = tf.placeholder(tf.int32, [None, 2], name="actions_num")
#         self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
#         # hidden layers
#         h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
#         # softmax layer
#         self.softmax_input = tf.matmul(h_layer, W2) + b2
#         # softmax output
#         self.all_act_prob = tf.nn.softmax(self.softmax_input, name='act_prob')
#
#         self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_input,
#                                                                     labels=self.tf_acts)
#         self.exp = tf.reduce_mean(self.neg_log_prob * self.td_error)
#
#         # 这里需要最大化当前策略的价值，因此需要最大化self.exp,即最小化-self.exp
#         self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(-self.exp)
#
#     def weight_variable(self, shape):
#         initial = tf.truncated_normal(shape)
#         return tf.Variable(initial)
#
#     def bias_variable(self, shape):
#         initial = tf.constant(0.01, shape=shape)
#         return tf.Variable(initial)
#
#     def choose_action(self, observation):
#         prob_weights = self.session.run(self.all_act_prob, feed_dict={self.state_input: observation[np.newaxis, :]})
#         action = np.random.choice(range(prob_weights.shape[1]),
#                                   p=prob_weights.ravel())  # select action w.r.t the actions prob
#         return action
#
#     def learn(self, state, action, td_error):
#         s = state[np.newaxis, :]
#         one_hot_action = np.zeros(self.action_dim)
#         one_hot_action[action] = 1
#         a = one_hot_action[np.newaxis, :]
#         # train on episode
#         self.session.run(self.train_op, feed_dict={
#             self.state_input: s,
#             self.tf_acts: a,
#             self.td_error: td_error,
#         })
#
#
# EPSILON = 0.01  # final value of epsilon
# REPLAY_SIZE = 10000  # experience replay buffer size
# BATCH_SIZE = 32  # size of minibatch
# REPLACE_TARGET_FREQ = 10  # frequency to update target Q network
#
#
# class Critic():
#     def __init__(self, env, sess):
#         # init some parameters
#         self.time_step = 0
#         self.epsilon = EPSILON
#         self.state_dim = env.observation_space.shape[0]
#         self.action_dim = env.action_space.n
#
#         self.create_Q_network()
#         self.create_training_method()
#
#         # Init session
#         self.session = sess
#         self.session.run(tf.global_variables_initializer())
#
#     def create_Q_network(self):
#         # network weights
#         W1q = self.weight_variable([self.state_dim, 20])
#         b1q = self.bias_variable([20])
#         W2q = self.weight_variable([20, 1])
#         b2q = self.bias_variable([1])
#         self.state_input = tf.placeholder(tf.float32, [1, self.state_dim], "state")
#         # hidden layers
#         h_layerq = tf.nn.relu(tf.matmul(self.state_input, W1q) + b1q)
#         # Q Value layer
#         self.Q_value = tf.matmul(h_layerq, W2q) + b2q
#
#     def create_training_method(self):
#         self.next_value = tf.placeholder(tf.float32, [1, 1], "v_next")
#         self.reward = tf.placeholder(tf.float32, None, 'reward')
#
#         with tf.variable_scope('squared_TD_error'):
#             self.td_error = self.reward + GAMMA * self.next_value - self.Q_value
#             self.loss = tf.square(self.td_error)
#         with tf.variable_scope('train'):
#             self.train_op = tf.train.AdamOptimizer(self.epsilon).minimize(self.loss)
#
#     def train_Q_network(self, state, reward, next_state):
#         s, s_ = state[np.newaxis, :], next_state[np.newaxis, :]
#         v_ = self.session.run(self.Q_value, {self.state_input: s_})
#         td_error, _ = self.session.run([self.td_error, self.train_op],
#                                        {self.state_input: s, self.next_value: v_, self.reward: reward})
#         return td_error
#
#     def weight_variable(self, shape):
#         initial = tf.truncated_normal(shape)
#         return tf.Variable(initial)
#
#     def bias_variable(self, shape):
#         initial = tf.constant(0.01, shape=shape)
#         return tf.Variable(initial)
#
#
# # Hyper Parameters
# ENV_NAME = 'CartPole-v0'
# EPISODE = 3000  # Episode limitation
# STEP = 3000  # Step limitation in an episode
# TEST = 10  # The number of experiment test every 100 episode
#
#
# def main():
#     # initialize OpenAI Gym env and dqn agent
#     sess = tf.InteractiveSession()
#     env = gym.make(ENV_NAME)
#     actor = Actor(env, sess)
#     critic = Critic(env, sess)
#
#     for episode in range(EPISODE):
#         # initialize task
#         state = env.reset()
#         # Train
#         for step in range(STEP):
#             action = actor.choose_action(state)  # e-greedy action for train
#             next_state, reward, done, _ = env.step(action)
#             td_error = critic.train_Q_network(state, reward, next_state)  # gradient = grad[r + gamma * V(s_) - V(s)]
#             actor.learn(state, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
#             state = next_state
#             if done:
#                 break
#
#         # Test every 100 episodes
#         if episode % 100 == 0:
#             total_reward = 0
#             for i in range(TEST):
#                 state = env.reset()
#                 for j in range(STEP):
#                     env.render()
#                     action = actor.choose_action(state)  # direct action for test
#                     state, reward, done, _ = env.step(action)
#                     total_reward += reward
#                     if done:
#                         break
#             ave_reward = total_reward / TEST
#             print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
#
#
# if __name__ == '__main__':
#     main()
#
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import copy
import gym

model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(3, )),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
])
model1.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.losses.mse,
)

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(3, )),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
])
model2.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.losses.mse,
)
print(model1.predict(np.array([[1, 2, 3]])))
print(model2.predict(np.array([[1, 2, 3]])))
model2.set_weights(model1.get_weights())
print(model2.predict(np.array([[1, 2, 3]])))
print(model2.get_weights())

