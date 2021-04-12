from Lib import *
import numpy as np
import pandas as pd
import time
import tkinter as tk
import threading

class Agent(object):
    def __init__(self, env, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.env = env
        self.learning_rate, self.reward_decay, self.e_greddy = learning_rate, reward_decay, e_greedy
        self._idxs = {}
        self.tables = []
        q_table = pd.DataFrame(columns=range(len(self.env.ACTION)), dtype=np.float64)
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
                        [0] * len(self.env.ACTION),
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
            return np.random.choice(range(len(self.env.ACTION)))
        else:
            self.check_state_exist(self.idx)
            line = self.q_table.loc[self.idx, :]
            return np.random.choice(line[line == line.max()].index)

class Agent_Q(Agent):
    __name__ = 'Q'
    def choose_action(self, isOpponent=False):
        return self._select_max_table_idx(isOpponent)

    def _select_max_table_idx(self, isOpponent):
        if np.random.uniform() >= self.e_greddy:
            return np.random.choice(range(len(self.env.ACTION)))
        elif not isOpponent:
            self.check_state_exist(self.idx)
            line = self.q_table.loc[self.idx, :]
            return np.random.choice(line[line == line.max()].index)
        else:
            self.check_state_exist(self.idx)
            line = self.q_table.loc[self.idx, :]
            return np.random.choice(line[line == line.min()].index)


    def learn(self, action, reward, lastIdx):
        self.check_state_exist(self.idx)
        self.check_state_exist(lastIdx)
        q_predict = self.q_table.loc[lastIdx, action]
        q_target = reward + self.reward_decay * self.q_table.loc[self.idx, :].max()
        self.q_table.loc[lastIdx, action] += self.learning_rate * (q_target - q_predict)

class Agent_Sarsa(Agent):
    __name__ = 'Sarsa'
    def __init__(self, *args, **kwargs):
        super(Agent_Sarsa, self).__init__(*args, **kwargs)
        eligibility_trace = pd.DataFrame(columns=range(len(self.env.ACTION)), dtype=np.float64)
        self.tables.append(eligibility_trace)
        self.next_action = np.random.choice(range(len(self.env.ACTION)))

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
    def buildMap(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def getIdx(self):
        raise NotImplementedError

    def evaluate(self, action):
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

    def evaluate(self, action):
        tx, ty = self.agent.x + Maze.ACTION[action][0], self.agent.y + Maze.ACTION[action][1]
        if tx < 0 or ty < 0 or tx >= self.width or ty >= self.height:
            return None, -1, True
        else:
            return (tx, ty), Maze.TREASURE.get(self.map[ty][tx]), Maze.ISEND.get(self.map[ty][tx])

    def step(self, next_state):
        x, y = next_state
        self.map[self.agent.y][self.agent.x], self.map[y][x] = Maze.TYPE.GROUND, Maze.TYPE.AGENT
        self._updateGrid(self.agent.x, self.agent.y)
        self._updateGrid(x, y)
        self.agent.x, self.agent.y = x, y

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

class Jing(Env):
    class TYPE(object):
        EMPTY = 0
        BLACK = 1
        RED = 2

    class STATE(object):
        NONE = 0
        SUCCESS = 1
        FAILED = 2
        DRAW = 3

    TYPE2COLOR = {
        TYPE.EMPTY: 'white',
        TYPE.BLACK: 'black',
        TYPE.RED: 'red',
    }
    SIZE = 3
    ACTION = [[j, i] for i in range(3) for j in range(3)]
    MAZE_W, MAZE_H = 40, 40
    GRID_W, GRID_H = 30, 30
    def buildMap(self):
        self.agent = Agent_Q(self)
        self.map = [[Jing.TYPE.EMPTY] * Jing.SIZE for i in range(Jing.SIZE)]
        self.idx = 0
        self.player = Jing.TYPE.BLACK
        self._createMap()

    def reset(self):
        self.map = [[Jing.TYPE.EMPTY] * Jing.SIZE for i in range(Jing.SIZE)]
        self.idx = 0
        self.player = Jing.TYPE.BLACK
        self._createMap()

    def getIdx(self):
        return self.idx

    def step(self, next_state):
        if next_state is None:
            return
        x, y = next_state
        self.map[y][x] = self.player
        self.idx += 3 ** (y * Jing.SIZE + x) * self.player
        self.player = Jing.TYPE.BLACK if self.player == Jing.TYPE.RED else Jing.TYPE.RED
        self._updateGrid(x, y)

    def evaluate(self, action):
        x, y = Jing.ACTION[action][0], Jing.ACTION[action][1]
        if self.map[y][x] == Jing.TYPE.EMPTY:
            self.map[y][x] = self.player
            state = self.checkState()
            if state == Jing.STATE.NONE:
                ret = (x, y), 0, False
            elif state == Jing.STATE.DRAW:
                ret = (x, y), 0, True
            elif state == Jing.STATE.SUCCESS:
                ret = (x, y), 1, True
            else:
                ret = (x, y), -1, True
            self.map[y][x] = Jing.TYPE.EMPTY
            return ret
        else:
            return None, -1 if self.player == Jing.TYPE.BLACK else 1, True

    def checkState(self):
        def checkFull():
            for i in range(Jing.SIZE):
                for j in range(Jing.SIZE):
                    if self.map[i][j] == Jing.TYPE.EMPTY:
                        return False
            return True

        def checkLine():
            for i in range(Jing.SIZE):
                c = self.map[i][0]
                if c == Jing.TYPE.EMPTY:
                    continue
                for j in range(1, Jing.SIZE):
                    if self.map[i][j] != c:
                        break
                else:
                    return c
            for j in range(Jing.SIZE):
                c = self.map[0][j]
                if c == Jing.TYPE.EMPTY:
                    continue
                for i in range(1, Jing.SIZE):
                    if self.map[i][j] != c:
                        break
                else:
                    return c
            return None

        line = checkLine()
        if line is not None:
            return Jing.STATE.SUCCESS if line == Jing.TYPE.BLACK else Jing.STATE.FAILED
        if checkFull():
            return Jing.STATE.DRAW
        return Jing.STATE.NONE

    def _createMap(self):
        if not hasattr(self, 'canvas'):
            self.canvas = tk.Canvas(self, bg='white', height=Jing.MAZE_H * Jing.SIZE, width=Jing.MAZE_W * Jing.SIZE)
            # create lines
            right, bottom = Jing.MAZE_W * Jing.SIZE, Jing.MAZE_H * Jing.SIZE
            for c in range(0, right, Jing.MAZE_W):
                self.canvas.create_line(c, 0, c, bottom)
            for r in range(0, bottom, Jing.MAZE_H):
                self.canvas.create_line(0, r, right, r)
            self.canvas.pack()
        else:
            for r in self.grids:
                for c in r:
                    self.canvas.delete(c)
        # create grids
        self.grids = []
        for i in range(Jing.SIZE):
            self.grids.append([0] * Jing.SIZE)
            for j in range(Jing.SIZE):
                self._updateGrid(j, i)

    def _updateGrid(self, x, y):
        if self.grids[y][x]:
            self.canvas.delete(self.grids[y][x])
        self.grids[y][x] = self.canvas.create_rectangle(
            (x + 0.5) * Jing.MAZE_W - 0.5 * Jing.GRID_W, (y + 0.5) * Jing.MAZE_H - 0.5 * Jing.GRID_H,
            (x + 0.5) * Jing.MAZE_W + 0.5 * Jing.GRID_W, (y + 0.5) * Jing.MAZE_H + 0.5 * Jing.GRID_H,
            fill=Jing.TYPE2COLOR.get(self.map[y][x])
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
            terminate = False
            startTime = time.time()
            step = 0
            while True:
                action = self.env.agent.choose_action()
                next_state, reward, ter = self.env.evaluate(action)
                if next_state is None:
                    continue
                lastIdx = self.env.agent.idx
                self.env.step(next_state)
                self.env.agent.learn(action, reward, lastIdx)
                terminate = terminate or ter
                isSuccess = reward == 1
                step += 1
                self.showProcess and self.env.render(0.5)
                if terminate:
                    break
            print('episode {}, result {}, takes {} steps {} second'.format(episode, isSuccess, step, time.time() - startTime))

    def createEnv(self):
        raise NotImplementedError

class ThreadMaze(ThreadBase):
    def createEnv(self):
        self.env = Maze()
        self.env.buildMap(readFile(getDataFilePath('Q&S-SimpleMaze.txt')))

class ThreadJing(ThreadBase):
    def createEnv(self):
        self.env = Jing()
        self.env.buildMap()

if __name__ == '__main__':
    # thread = ThreadMaze(showProcess=False)
    thread = ThreadJing(showProcess=True)
    thread.start()
    while True:
        thread.showProcess = input().strip() != '0'