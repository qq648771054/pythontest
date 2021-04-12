import numpy as np
import pandas as pd
import time
import tkinter as tk

class Agent(object):
    def __init__(self, map, x, y, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.map = map
        self.x, self.y = x, y
        self.learning_rate, self.reward_decay, self.e_greddy = learning_rate, reward_decay, e_greedy
        self._createTable()
        self.check_state_exist(self.idx)

    @property
    def idx(self):
        return self.map.width * self.y + self.x

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.map.ACTION),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self):
        raise NotImplementedError

    def learn(self, action, x, y, reward):
        raise NotImplementedError

    def _createTable(self):
        self.q_table = pd.DataFrame(columns=range(len(self.map.ACTION)), dtype=np.float64)

    def _select_max_table_idx(self):
        if np.random.uniform() >= self.e_greddy:
            return np.random.choice(range(len(self.map.ACTION)))
        else:
            line = self.q_table.loc[self.idx, :]
            return np.random.choice(line[line == line.max()].index)

class Agent_Q(Agent):
    __name__ = 'Q'
    def choose_action(self):
        return self._select_max_table_idx()

    def learn(self, action, x, y, reward):
        lastIdx = self.idx
        self.map.step(self, x, y)
        nextIdx = self.idx
        q_predict = self.q_table.loc[lastIdx, action]
        self.check_state_exist(nextIdx)
        q_target = reward + self.reward_decay * self.q_table.loc[nextIdx, :].max()
        self.q_table.loc[lastIdx, action] += self.learning_rate * (q_target - q_predict)

class Agent_Sarsa(Agent):
    __name__ = 'Sarsa'
    def __init__(self, *args, **kwargs):
        super(Agent_Sarsa, self).__init__(*args, **kwargs)
        self.next_action = np.random.choice(range(len(self.map.ACTION)))

    def choose_action(self):
        return self.next_action

    def learn(self, action, x, y, reward):
        lastIdx = self.idx
        self.map.step(self, x, y)
        nextIdx = self.idx
        q_predict = self.q_table.loc[lastIdx, action]
        self.next_action = self._select_max_table_idx()
        q_target = reward + self.reward_decay * self.q_table.loc[nextIdx, self.next_action]
        self.q_table.loc[lastIdx, action] += self.learning_rate * (q_target - q_predict)

class Agent_Sarsa_Lambda(Agent_Sarsa):
    __name__ = 'Sarsa_Lambda'
    def __init__(self, *args, _lambda=0.9,  **kwargs):
        super(Agent_Sarsa_Lambda, self).__init__(*args, **kwargs)
        self._lambda = _lambda

    def _createTable(self):
        super(Agent_Sarsa_Lambda, self)._createTable()
        self.eligibility_trace = pd.DataFrame(columns=range(len(self.map.ACTION)), dtype=np.float64)

    def check_state_exist(self, state):
        super(Agent_Sarsa_Lambda, self).check_state_exist(state)
        if state not in self.eligibility_trace.index:
            # append new state to q table
            self.eligibility_trace = self.eligibility_trace.append(
                pd.Series(
                    [0] * len(self.map.ACTION),
                    index=self.eligibility_trace.columns,
                    name=state,
                )
            )

    def learn(self, action, x, y, reward):
        lastIdx = self.idx
        self.map.step(self, x, y)
        nextIdx = self.idx
        q_predict = self.q_table.loc[lastIdx, action]
        self.check_state_exist(nextIdx)
        self.next_action = self._select_max_table_idx()
        q_target = reward + self.reward_decay * self.q_table.loc[nextIdx, self.next_action]
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
        TYPE.GROUND: 0,
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

    # def __init__(self, *args, **kwargs):
    #     super(Env).__init__(*args, **kwargs)
    #     self.canvas = None

    def buildMap(self, map):
        map = [m.strip() for m in map.split('\n') if len(m.strip()) > 0]
        self.rawMap = map
        self.width = len(map[0])
        self.height = len(map)
        self.map = [[x for x in m] for m in map]
        self.agents = []
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] in Env.AGENT2TYPE:
                    self.agents.append(Env.AGENT2TYPE.get(self.map[i][j])(self, j, i))
                    self.map[i][j] = Env.TYPE.AGENT
        self._createMap()

    def reset(self):
        map = self.rawMap
        self.map = [[x for x in m] for m in map]
        agentidx = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] in Env.AGENT2TYPE:
                    agent = self.agents[agentidx]
                    agent.x, agent.y = j, i
                    agentidx += 1
                    self.map[i][j] = Env.TYPE.AGENT
        self._createMap()

    def evaluate(self, agent, action):
        tx, ty = agent.x + Env.ACTION[action][0], agent.y + Env.ACTION[action][1]
        if tx < 0 or ty < 0 or tx >= self.width or ty >= self.height:
            return agent.x, agent.y, -1, True
        else:
            return tx, ty, Env.TREASURE.get(self.map[ty][tx]), Env.ISEND.get(self.map[ty][tx])

    def step(self, agent, x, y):
        self.map[agent.y][agent.x], self.map[y][x] = Env.TYPE.GROUND, Env.TYPE.AGENT
        self._updateGrid(agent.x, agent.y)
        self._updateGrid(x, y)
        agent.x, agent.y = x, y

    def render(self, sleepTime=0.5):
        self.update()
        time.sleep(sleepTime)

    def _createMap(self):
        if not hasattr(self, 'canvas'):
            self.canvas = tk.Canvas(self, bg='white', height=Env.MAZE_H * self.height, width=Env.MAZE_W * self.width)
            # create lines
            right, bottom = Env.MAZE_W * self.width, Env.MAZE_H * self.height
            for c in range(0, right, Env.MAZE_W):
                self.canvas.create_line(c, 0, c, bottom)
            for r in range(0, bottom, Env.MAZE_H):
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
            (x + 0.5) * Env.MAZE_W - 0.5 * Env.GRID_W, (y + 0.5) * Env.MAZE_H - 0.5 * Env.GRID_H,
            (x + 0.5) * Env.MAZE_W + 0.5 * Env.GRID_W, (y + 0.5) * Env.MAZE_H + 0.5 * Env.GRID_H,
            fill=Env.TYPE2COLOR.get(self.map[y][x])
        )

def run():
    MAP = '''
.....
.L...
..#..
.#e..
.....
    '''
    map = Env()
    map.buildMap(MAP)
    agents = map.agents
    epoch = 0
    while True:
        map.reset()
        map.render()
        epoch += 1
        terminate = False
        while True:
            for agent in agents:
                action = agent.choose_action()
                x, y, reward, ter = map.evaluate(agent, action)
                agent.learn(action, x, y, reward)
                terminate = terminate or ter
            map.render(0.05)
            if terminate:
                break

if __name__ == '__main__':
    run()