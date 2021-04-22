from lib import *
import Thread
import envoriment

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.object)
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Agent_DQN(object):
    def __init__(self, env):
        self.env = env
        self.model = self._buildModel(self.env.stateShape, self.env.actionLen)
        self.bak_model = copyModel(self.model)
        self.learnTime = 0
        self.learnCycle = 10
        self.memory_size = 1000
        self.memory = Memory(self.memory_size)

    def save_exp(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def learn(self, reward_decay=0.9, batch_size=32):
        self.learnTime += 1
        if self.learnTime % self.learnCycle == 0:
            self.bak_model = copyModel(self.model)
        idxs, memorys = self.memory.sample(batch_size)
        xs, ys, deltas = [], [], []
        for i, memory in zip(idxs, memorys):
            state, action, reward, next_state, done = memory
            if done:
                q_target = reward
            else:
                idx = self.model.predict(addAixs(next_state))[0].argmax()
                q_target = reward + reward_decay * self.bak_model.predict(addAixs(next_state))[0][idx]
            q_predict = self.model.predict(addAixs(state))
            deltas.append(abs(q_target - q_predict[0][action]))
            q_predict[0][action] = q_target
            xs.append(state)
            ys.append(q_predict[0])
        self.model.fit(np.array(xs), np.array(ys), epochs=1, verbose=0)
        self.memory.batch_update(idxs, np.array(deltas))

    def choose_action(self, state, e_greddy=0.9):
        if np.random.rand() >= e_greddy:
            return np.random.randint(0, self.env.actionLen)
        else:
            act_values = self.model.predict(addAixs(state))
            return np.argmax(act_values[0])

    def _buildModel(self, stateShape, actionLen, learning_rate=0.01):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation='relu', input_shape=stateShape),
            tf.keras.layers.Dense(actionLen, activation='linear')
        ])
        model.compile(
            loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate)
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
    # thread = ThreadMaze(showProcess=False, savePath=getDataFilePath('dqn_maze_record.h5'))
    thread = ThreadCartPole(Agent_DQN, showProcess=True)
    thread.start()
    while True:
        cmd = input()
        if cmd == '0':
            thread.showProcess = False
        else:
            thread.showProcess = True

