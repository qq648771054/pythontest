from lib import *
import gym
import tkinter as tk

class EnvNN(object):
    def __init__(self, agentType):
        self.agent = agentType(self)

    @property
    def actionLen(self):
        raise NotImplementedError

    @property
    def stateShape(self):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

class EnvOpenAI(EnvNN):
    def __init__(self, env, agentType):
        self._env = env
        super(EnvOpenAI, self).__init__(agentType)

    @property
    def actionLen(self):
        return self._env.action_space.n

    @property
    def stateShape(self):
        return self._env.observation_space.shape

    def reset(self):
        return self._env.reset()

    def render(self):
        self._env.render()

    def step(self, action):
        return self._env.step(action)[: 3]

class CartPole_v0(EnvOpenAI):
    def __init__(self, agentType):
        super(CartPole_v0, self).__init__(gym.make('CartPole-v0'), agentType)

    def step(self, action):
        next_state, reward, done = super(CartPole_v0, self).step(action)
        x, x_dot, theta, theta_dot = next_state
        env = self._env
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return next_state, reward, done





class EnvTk(EnvNN, tk.Tk):
    ACTION = []

    def __init__(self, agentType):
        tk.Tk.__init__(self)
        EnvNN.__init__(self, agentType)

    @property
    def actionLen(self):
        return len(self.ACTION)

    def render(self):
        self.update()

class Maze(EnvTk):
    class TYPE:
        GROUND = 0
        START = 1
        END = 2
        TRAP = 3
    TYPE2ID = {
        '.': TYPE.GROUND,
        's': TYPE.START,
        'e': TYPE.END,
        '#': TYPE.TRAP,
    }
    TYPE2REWARD = {
        TYPE.GROUND: 0,
        TYPE.START: 0,
        TYPE.END: 1,
        TYPE.TRAP: -1
    }
    TYPE2DONE = {
        TYPE.GROUND: False,
        TYPE.START: False,
        TYPE.END: True,
        TYPE.TRAP: True
    }
    TYPE2COLOR = {
        TYPE.GROUND: 'white',
        TYPE.START: 'black',
        TYPE.END: 'green',
        TYPE.TRAP: 'red'
    }

    ACTION = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    MAZE_W, MAZE_H = 30, 30
    GRID_W, GRID_H = 25, 25

    def __init__(self, width, height, agentType):
        self.width, self.height = width, height
        super(Maze, self).__init__(agentType)

    @property
    def stateShape(self):
        return self.height, self.width

    def reset(self, mapData):
        map = [m.strip() for m in mapData.split('\n') if len(m.strip()) > 0]
        self.map = []
        for i in range(self.height):
            self.map.append([])
            for j in range(self.width):
                self.map[i].append(self.TYPE2ID[map[i][j]])
                if self.map[i][j] == self.TYPE.START:
                    self.x, self.y = j, i
        self._createMap()
        return np.array(self.map)

    def step(self, action):
        x, y = self.x + self.ACTION[action][0], self.y + self.ACTION[action][1]
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return np.array(self.map), -1, True
        else:
            nextPath = self.map[y][x]
            self.map[y][x], self.map[self.y][self.x] = self.TYPE.START, self.TYPE.GROUND
            self._updateGrid(x, y)
            self._updateGrid(self.x, self.y)
            self.y, self.x = y, x
            reward, done = self.TYPE2REWARD[nextPath], self.TYPE2DONE[nextPath]
            return np.array(self.map), reward, done

    def _createMap(self):
        if not hasattr(self, 'canvas'):
            self.canvas = tk.Canvas(self, bg='white', height=self.MAZE_H * self.height, width=self.MAZE_W * self.width)
            # create lines
            right, bottom = self.MAZE_W * self.width, self.MAZE_H * self.height
            for c in range(0, right, self.MAZE_W):
                self.canvas.create_line(c, 0, c, bottom)
            for r in range(0, bottom, self.MAZE_H):
                self.canvas.create_line(0, r, right, r)
            self.canvas.pack()
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
            (x + 0.5) * self.MAZE_W - 0.5 * self.GRID_W, (y + 0.5) * self.MAZE_H - 0.5 * self.GRID_H,
            (x + 0.5) * self.MAZE_W + 0.5 * self.GRID_W, (y + 0.5) * self.MAZE_H + 0.5 * self.GRID_H,
            fill=self.TYPE2COLOR[self.map[y][x]]
        )


class Jing(tk.Tk):
    def __init__(self, agentType):
        super(Jing, self).__init__()
        self.agent = agentType(self)

    MAZE_W, MAZE_H = 40, 40
    GRID_W, GRID_H = 30, 30

    TYPE2COLOR = {
        0: 'white',
        1: 'black',
        2: 'red'
    }

    SIZE = 3
    CLASS_SIZE = 3
    ACTION_SIZE = SIZE * SIZE
    STATE_SIZE = CLASS_SIZE ** ACTION_SIZE

    def validActions(self, idx):
        res = []
        for i in range(self.SIZE * self.SIZE):
            if idx % self.CLASS_SIZE == 0:
                res.append(i)
            idx //= self.CLASS_SIZE
        return res

    def map2Idx(self, map):
        idx = 0
        for i in range(self.SIZE - 1, -1, -1):
            for j in range(self.SIZE - 1, -1, -1):
                idx = idx * self.CLASS_SIZE + map[i][j]
        return idx

    def idx2Map(self, idx):
        map = [[0] * self.SIZE for i in range(self.SIZE)]
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                map[i][j] = idx % self.CLASS_SIZE
                idx //= self.CLASS_SIZE
        return map

    def flipIdx(self, state):
        ids = []
        while state:
            p = state % self.CLASS_SIZE
            ids.append(self._otherPlayer(p) if p else 0)
            state //= self.CLASS_SIZE
        idx = 0
        for i in reversed(ids):
            idx = idx * self.CLASS_SIZE + i
        return idx

    def _otherPlayer(self, player):
        return 1 if player == 2 else 2

    def reset(self):
        self.map = [[0] * self.SIZE for i in range(self.SIZE)]
        self.player = 2
        self._createMap()
        return self.map2Idx(self.map)

    def step(self, action):
        x, y = action % self.SIZE, action // self.SIZE
        self.player = self._otherPlayer(self.player)
        self.map[y][x] = self.player
        next_state = self.map2Idx(self.map)
        next_state = next_state if self.player == 1 else self.flipIdx(next_state)
        self._updateGrid(x, y)
        return next_state, self.player, self.getWiner()

    def getNextState(self, state, action):
        state = self.flipIdx(state)
        state += self.CLASS_SIZE ** action
        return state

    def getWiner(self):
        def checkLine():
            for i in range(self.SIZE):
                c = self.map[i][0]
                if not c: continue
                for j in range(1, self.SIZE):
                    if self.map[i][j] != c:
                        break
                else:
                    return c
            return None
        def checkColumn():
            for j in range(self.SIZE):
                c = self.map[0][j]
                if not c: continue
                for i in range(1, self.SIZE):
                    if self.map[i][j] != c:
                        break
                else:
                    return c
            return None
        def checkDiagonal():
            c = self.map[0][0]
            if c:
                for i in range(1, self.SIZE):
                    if self.map[i][i] != c:
                        break
                else:
                    return c
            c = self.map[0][self.SIZE - 1]
            if c:
                for i in range(1, self.SIZE):
                    if self.map[i][self.SIZE - i - 1] != c:
                        break
                else:
                    return c
            return None
        def checkFull():
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    if self.map[i][j] == 0:
                        return False
            return True
        winer = checkLine() or checkColumn() or checkDiagonal()
        if winer:
            return winer
        elif checkFull():
            return 0
        else:
            return None

    def _createMap(self):
        if not hasattr(self, 'canvas'):
            self.canvas = tk.Canvas(self, bg='white', height=self.MAZE_H * self.SIZE, width=self.MAZE_W * self.SIZE)
            # create lines
            right, bottom = self.MAZE_W * self.SIZE, self.MAZE_H * self.SIZE
            for c in range(0, right, self.MAZE_W):
                self.canvas.create_line(c, 0, c, bottom)
            for r in range(0, bottom, self.MAZE_H):
                self.canvas.create_line(0, r, right, r)
            self.canvas.pack()
        # create grids
        self.grids = []
        for i in range(self.SIZE):
            self.grids.append([0] * self.SIZE)
            for j in range(self.SIZE):
                self._updateGrid(j, i)

    def _updateGrid(self, x, y):
        if self.grids[y][x]:
            self.canvas.delete(self.grids[y][x])
        self.grids[y][x] = self.canvas.create_rectangle(
            (x + 0.5) * self.MAZE_W - 0.5 * self.GRID_W, (y + 0.5) * self.MAZE_H - 0.5 * self.GRID_H,
            (x + 0.5) * self.MAZE_W + 0.5 * self.GRID_W, (y + 0.5) * self.MAZE_H + 0.5 * self.GRID_H,
            fill=self.TYPE2COLOR.get(self.map[y][x])
        )

    def render(self):
        self.update()