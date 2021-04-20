from lib import *
import time
import tkinter as tk
import threading

class Agent_Single(object):
    def __init__(self, env, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.env = env
        self.learning_rate, self.reward_decay, self.e_greddy = learning_rate, reward_decay, e_greedy
        self._idxs = {}
        self.tables = []
        q_table = pd.DataFrame(columns=range(self.env.actionLen), dtype=np.float64)
        self.tables.append(q_table)

    @property
    def idx(self):
        return self.env.getIdx()

    @property
    def q_table(self):
        return self.tables[0]

    @q_table.setter
    def q_table(self, t):
        self.tables[0] = t

    def check_state_exist(self, state):
        if state not in self._idxs:
            self._idxs[state] = True
            for i, table in enumerate(self.tables):
                self.tables[i] = table.append(
                    pd.Series(
                        [0] * self.env.actionLen,
                        index=table.columns,
                        name=state,
                    )
                )

    def choose_action(self):
        raise NotImplementedError

    def learn(self, action, reward, lastIdx):
        raise NotImplementedError

    def _select_max_table_idx(self):
        if np.random.uniform() >= self.e_greddy:
            return np.random.choice(range(self.env.actionLen))
        else:
            self.check_state_exist(self.idx)
            line = self.q_table.loc[self.idx, :]
            return np.random.choice(line[line == line.max()].index)

class Agent_Q(Agent_Single):
    __name__ = 'Q'
    def choose_action(self):
        return self._select_max_table_idx()

    def learn(self, action, reward, lastIdx):
        self.check_state_exist(self.idx)
        self.check_state_exist(lastIdx)
        q_predict = self.q_table.loc[lastIdx, action]
        q_target = reward + self.reward_decay * self.q_table.loc[self.idx, :].max()
        self.q_table.loc[lastIdx, action] += self.learning_rate * (q_target - q_predict)

class Agent_Sarsa(Agent_Single):
    __name__ = 'Sarsa'
    def __init__(self, *args, **kwargs):
        super(Agent_Sarsa, self).__init__(*args, **kwargs)
        eligibility_trace = pd.DataFrame(columns=range(self.env.actionLen), dtype=np.float64)
        self.tables.append(eligibility_trace)
        self.next_action = np.random.choice(range(self.env.actionLen))

    @property
    def eligibility_trace(self):
        return self.tables[1]

    @eligibility_trace.setter
    def eligibility_trace(self,  t):
        self.tables[1] = t

    def choose_action(self):
        return self.next_action

    def learn(self, action, reward, lastIdx):
        self.check_state_exist(self.idx)
        self.check_state_exist(lastIdx)
        q_predict = self.q_table.loc[lastIdx, action]
        self.next_action = self._select_max_table_idx()
        q_target = reward + self.reward_decay * self.q_table.loc[self.idx, self.next_action]
        self.q_table.loc[lastIdx, action] += self.learning_rate * (q_target - q_predict)

class Agent_Sarsa_Lambda(Agent_Sarsa):
    __name__ = 'Sarsa_Lambda'
    def __init__(self, *args, _lambda=0.9,  **kwargs):
        super(Agent_Sarsa_Lambda, self).__init__(*args, **kwargs)
        self._lambda = _lambda

    def learn(self, action, reward, lastIdx):
        self.check_state_exist(self.idx)
        self.check_state_exist(lastIdx)
        q_predict = self.q_table.loc[lastIdx, action]
        self.next_action = self._select_max_table_idx()
        q_target = reward + self.reward_decay * self.q_table.loc[self.idx, self.next_action]
        error = q_target - q_predict
        self.eligibility_trace.loc[lastIdx, :] = 0
        self.eligibility_trace.loc[lastIdx, action] = 1
        self.q_table += self.learning_rate * error * self.eligibility_trace
        self.eligibility_trace *= self.reward_decay * self._lambda

class MyDict(dict):
    def get(self, item):
        if item in self:
            return self[item]
        else:
            return self['DEFAULT']

class Env(tk.Tk):
    ACTION = []

    @property
    def actionLen(self):
        return len(self.ACTION)

    def buildMap(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def getIdx(self):
        raise NotImplementedError

    def step(self, next_state):
        raise NotImplementedError

    def render(self, sleepTime=0.5):
        self.update()
        time.sleep(sleepTime)

class Maze(Env):
    class TYPE(object):
        AGENT = 'a'
        END = 'e'
        GROUND = '.'
        TRAP = '#'
        AGENT_Q = 'A'
        AGENT_SARSA = 'S'
        AGENT_SARSA_LAMBDA = 'L'

    TREASURE = MyDict({
        TYPE.AGENT: -1,
        TYPE.END: 1,
        TYPE.GROUND: -0.0001,
        TYPE.TRAP: -1,
        'DEFAULT': 0,
    })

    ISEND = MyDict({
        TYPE.AGENT: True,
        TYPE.END: True,
        TYPE.GROUND: False,
        TYPE.TRAP: True,
        'DEFAULT': False,
    })

    TYPE2COLOR = MyDict({
        TYPE.AGENT: 'black',
        TYPE.END: 'green',
        TYPE.GROUND: 'white',
        TYPE.TRAP: 'red',
        'DEFAULT': 'blue',
    })

    AGENT2TYPE = MyDict({
        TYPE.AGENT_Q: Agent_Q,
        TYPE.AGENT_SARSA: Agent_Sarsa,
        TYPE.AGENT_SARSA_LAMBDA: Agent_Sarsa_Lambda,
        'DEFAULT': Agent_Q
    })

    ACTION = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    MAZE_W, MAZE_H = 40, 40
    GRID_W, GRID_H = 30, 30

    def buildMap(self, map):
        map = [m.strip() for m in map.split('\n') if len(m.strip()) > 0]
        self.rawMap = map
        self.width = len(map[0])
        self.height = len(map)
        self.map = [[x for x in m] for m in map]
        self.agent = None
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] in Maze.AGENT2TYPE:
                    agent = Maze.AGENT2TYPE.get(self.map[i][j])(self)
                    agent.x, agent.y = j, i
                    self.agent = agent
                    self.map[i][j] = Maze.TYPE.AGENT
        self._createMap()

    def reset(self):
        map = self.rawMap
        self.map = [[x for x in m] for m in map]
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] in Maze.AGENT2TYPE:
                    agent = self.agent
                    agent.x, agent.y = j, i
                    self.map[i][j] = Maze.TYPE.AGENT
        self._createMap()

    def getIdx(self):
        return self.width * self.agent.y + self.agent.x

    def step(self, action):
        x, y = self.agent.x + Maze.ACTION[action][0], self.agent.y + Maze.ACTION[action][1]
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return (self.agent.x, self.agent.y), -1, False
        else:
            treasure, isend = Maze.TREASURE.get(self.map[y][x]), Maze.ISEND.get(self.map[y][x])
            self.map[self.agent.y][self.agent.x], self.map[y][x] = Maze.TYPE.GROUND, Maze.TYPE.AGENT
            self._updateGrid(self.agent.x, self.agent.y)
            self._updateGrid(x, y)
            self.agent.x, self.agent.y = x, y
            return (x, y), treasure, isend

    def _createMap(self):
        if not hasattr(self, 'canvas'):
            self.canvas = tk.Canvas(self, bg='white', height=Maze.MAZE_H * self.height, width=Maze.MAZE_W * self.width)
            # create lines
            right, bottom = Maze.MAZE_W * self.width, Maze.MAZE_H * self.height
            for c in range(0, right, Maze.MAZE_W):
                self.canvas.create_line(c, 0, c, bottom)
            for r in range(0, bottom, Maze.MAZE_H):
                self.canvas.create_line(0, r, right, r)
            self.canvas.pack()
        else:
            for r in self.grids:
                for c in r:
                    self.canvas.delete(c)
        # create grids
        self.grids = []
        for i in range(self.height):
            self.grids.append([0] * self.width)
            for j in range(self.width):
                self._updateGrid(j, i)

    def _updateGrid(self, x, y):
        if self.grids[y][x]:
            self.canvas.delete(self.grids[y][x])
        self.grids[y][x] = self.canvas.create_rectangle(
            (x + 0.5) * Maze.MAZE_W - 0.5 * Maze.GRID_W, (y + 0.5) * Maze.MAZE_H - 0.5 * Maze.GRID_H,
            (x + 0.5) * Maze.MAZE_W + 0.5 * Maze.GRID_W, (y + 0.5) * Maze.MAZE_H + 0.5 * Maze.GRID_H,
            fill=Maze.TYPE2COLOR.get(self.map[y][x])
        )

class ThreadBase(threading.Thread):
    def __init__(self, showProcess=True, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.showProcess = showProcess

    def run(self):
        self.createEnv()
        episode = 0
        while True:
            self.env.reset()
            self.showProcess and self.env.render()
            episode += 1
            startTime = time.time()
            step = 0
            while True:
                action = self.env.agent.choose_action()
                lastIdx = self.env.agent.idx
                next_state, reward, terminate = self.env.step(action)
                self.env.agent.learn(action, reward, lastIdx)
                step += 1
                self.showProcess and self.env.render(0.5)
                if terminate:
                    break
            print('episode {}, result {}, takes {} steps {} second'.format(episode, reward == 1, step, time.time() - startTime))

    def createEnv(self):
        raise NotImplementedError

class ThreadMaze(ThreadBase):
    def createEnv(self):
        self.env = Maze()
        self.env.buildMap(readFile(getDataFilePath('Q&S-SimpleMaze.txt')))

if __name__ == '__main__':
    # thread = ThreadMaze(showProcess=False)
    thread = ThreadMaze(showProcess=False)
    thread.start()
    while True:
        thread.showProcess = input().strip() != '0'