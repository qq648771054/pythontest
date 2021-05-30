from lib import *
import tkinter as tk
import copy

class Gobang(tk.Tk):

    MAZE_W, MAZE_H = 40, 40
    GRID_W, GRID_H = 30, 30

    TYPE2COLOR = {
        0: 'white',
        1: 'black',
        2: 'red'
    }

    SIZE = 15
    CLASS_SIZE = 3
    ACTION_SIZE = SIZE * SIZE
    STATE_SIZE = CLASS_SIZE ** ACTION_SIZE
    WIN_LENGTH = 5

    def __init__(self, onGridClick=None):
        super(Gobang, self).__init__()
        self.calSameShape()
        self.onGridClick = onGridClick

    def reshape(self, size=15, winLength=5):
        Gobang.SIZE = size
        Gobang.CLASS_SIZE = 3
        Gobang.ACTION_SIZE = Gobang.SIZE * Gobang.SIZE
        Gobang.STATE_SIZE = Gobang.CLASS_SIZE ** Gobang.ACTION_SIZE
        Gobang.WIN_LENGTH = winLength
        self.calSameShape()

    def validActions(self, state=None, player=None):
        if state is None:
            state = self.map
        direct = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]

        def extend(x, y, dx, dy):
            tx, ty = x + dx, y + dy
            c = state[ty][tx] if 0 <= tx < self.SIZE and 0 <= ty < self.SIZE else 0
            if c != 0:
                r = 1
                while True:
                    tx, ty = tx + dx, ty + dy
                    if 0 <= tx < self.SIZE and 0 <= ty < self.SIZE and state[ty][tx] == c:
                        r += 1
                    else:
                        break
                return r, c
            else:
                return 0, 0

        all = []
        m1, m2 = set(), set()
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if state[i][j] == 0:
                    idx = i * self.SIZE + j
                    all.append(idx)
                    d = [extend(j, i, direct[k][0], direct[k][1]) for k in range(8)]
                    for k in range(0, 8, 2):
                        if d[k][1] == d[k + 1][1]:
                            l = d[k][0] + d[k + 1][0]
                            c = d[k][1]
                        else:
                            l, c = 0, 0
                            if d[k][0] >= d[k + 1][0]:
                                l = d[k][0]
                                c |= d[k][1]
                            if d[k + 1][0] >= d[k][0]:
                                l = d[k + 1][0]
                                c |= d[k + 1][1]
                        if l >= self.WIN_LENGTH - 1:
                            if player is None or c & player:
                                m1.add(idx)
                            else:
                                m2.add(idx)
        if len(m1) > 0:
            return list(m1)
        elif len(m2) > 0:
            return list(m2)
        else:
            return all

    # def validActions(self, state=None, player=None):
    #     if state is None:
    #         state = self.map
    #     res = []
    #     for i in range(self.SIZE):
    #         for j in range(self.SIZE):
    #             if not state[i][j]:
    #                 res.append(i * self.SIZE + j)
    #     return res

    def flip(self, state):
        map = np.zeros((self.SIZE, self.SIZE), dtype=np.int)
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if state[i, j]:
                    map[i, j] = self._otherPlayer(state[i, j])
        return map

    def calSameShape(self):
        def rotate(board):
            return np.transpose(board)[::-1]

        def flipX(board):
            return board[:, ::-1]

        def flipY(board):
            return board[::-1, :]

        def sequence(board, *func):
            res = [board]
            for f in func:
                board = f(board)
                res.append(board)
            return res

        origin = np.array([[i + j for j in range(self.SIZE)] for i in range(0, self.ACTION_SIZE, self.SIZE)])
        self.sameShapes = []
        self.sameShapes.extend(sequence(origin, flipX, flipY, flipX))
        self.sameShapes.extend(sequence(origin, rotate, flipX, flipY, flipX))
        self.sameShapes.extend(sequence(origin, rotate, rotate, flipX, flipY, flipX))
        self.sameShapes = distinct(self.sameShapes, lambda a, b: (a == b).all())

    def board2Str(self, board):
        k = 0
        t = 0
        res = []
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                t = t * 4 + board[i][j]
                if k == 3:
                    res.append(chr(t))
                    k = 0
                    t = 0
                else:
                    k += 1
        if k > 0:
            res.append(chr(t))
        return ''.join(res)

    def _otherPlayer(self, player):
        return 1 if player == 2 else 2

    def reset(self):
        self.map = np.zeros((self.SIZE, self.SIZE), dtype=np.int)
        self.player = 2
        self._createMap()
        return copy.deepcopy(self.map)

    def step(self, action):
        x, y = action % self.SIZE, action // self.SIZE
        self.player = self._otherPlayer(self.player)
        self.map[y, x] = self.player
        next_state = copy.deepcopy(self.map) if self.player == 1 else self.flip(self.map)
        self._updateGrid(x, y)
        return next_state, self.player, self.getWiner(self.map)

    def getNextState(self, state, action):
        state = self.flip(state)
        state[action // self.SIZE, action % self.SIZE] = 1
        return state

    def extendState(self, state, prop):
        states, props = [], []
        for shape in self.sameShapes:
            s = [[0] * self.SIZE for i in range(self.SIZE)]
            p = [0] * (self.SIZE * self.SIZE)
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    idx = shape[i][j]
                    x, y = idx % self.SIZE, idx // self.SIZE
                    s[i][j] = state[y][x]
                    p[i * self.SIZE + j] = prop[y * self.SIZE + x]
            states.append(s)
            props.append(p)
        return states, props

    def getWiner(self, board):
        def checkLine():
            for i in range(self.SIZE):
                for j in range(self.SIZE - self.WIN_LENGTH + 1):
                    c = board[i][j]
                    if not c: continue
                    for k in range(1, self.WIN_LENGTH):
                        if board[i][j + k] != c:
                            break
                    else:
                        return c
            return None
        def checkColumn():
            for i in range(self.SIZE - self.WIN_LENGTH + 1):
                for j in range(self.SIZE):
                    c = board[i][j]
                    if not c: continue
                    for k in range(1, self.WIN_LENGTH):
                        if board[i + k][j] != c:
                            break
                    else:
                        return c
            return None
        def checkDiagonal():
            for i in range(self.SIZE - self.WIN_LENGTH + 1):
                for j in range(self.SIZE - self.WIN_LENGTH + 1):
                    c = board[i][j]
                    if not c: continue
                    for k in range(1, self.WIN_LENGTH):
                        if board[i + k][j + k] != c:
                            break
                    else:
                        return c
            for i in range(self.SIZE - self.WIN_LENGTH + 1):
                for j in range(self.WIN_LENGTH - 1, self.SIZE):
                    c = board[i][j]
                    if not c: continue
                    for k in range(1, self.WIN_LENGTH):
                        if board[i + k][j - k] != c:
                            break
                    else:
                        return c
            return None
        def checkFull():
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    if board[i][j] == 0:
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
            width, height = self.MAZE_W * self.SIZE, self.MAZE_H * self.SIZE
            right, bottom = width, height
            self.canvas = tk.Canvas(self, bg='white', height=height, width=width)
            # create lines
            for c in range(0, right, self.MAZE_W):
                self.canvas.create_line(c, 0, c, bottom)
            for r in range(0, bottom, self.MAZE_H):
                self.canvas.create_line(0, r, right, r)
            self.canvas.pack()
            self.canvas.bind('<Button-1>', self.onClick)
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

    def onClick(self, event):
        x, y = event.x // self.MAZE_W, event.y // self.MAZE_H
        self.onGridClick and self.onGridClick(x, y)
