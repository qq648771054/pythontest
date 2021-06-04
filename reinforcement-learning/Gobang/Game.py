from lib import *
import tkinter as tk
import copy

class Game(tk.Tk):

    MAZE_W, MAZE_H = 40, 40
    GRID_W, GRID_H = 30, 30

    TYPE2COLOR = {
        0: 'white',
        1: 'black',
        2: 'red'
    }

    def __init__(self, size=15, winLength=5, rendering=True, onGridClick=None):
        super(Game, self).__init__()
        self.classSize = 3
        self.size = size
        self.winLength = winLength
        self.actionSize = self.size ** 2
        self.stateSize = self.classSize ** self.actionSize
        self.calSameShape()
        self.rendering = rendering
        self.onGridClick = onGridClick

    def validActions(self, board=None):
        board = self.board if board is None else board
        player = 2
        direct = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]

        def extend(x, y, dx, dy):
            tx, ty = x + dx, y + dy
            c = board[ty][tx] if 0 <= tx < self.size and 0 <= ty < self.size else 0
            if c != 0:
                r = 1
                while True:
                    tx, ty = tx + dx, ty + dy
                    if 0 <= tx < self.size and 0 <= ty < self.size and board[ty][tx] == c:
                        r += 1
                    else:
                        break
                return r, c
            else:
                return 0, 0

        all = []
        m1, m2 = set(), set()
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    idx = i * self.size + j
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
                        if l >= self.winLength - 1:
                            if c & player:
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
    #         state = self.board
    #     res = []
    #     for i in range(self.size):
    #         for j in range(self.size):
    #             if not state[i][j]:
    #                 res.append(i * self.size + j)
    #     return res

    def flip(self, board=None):
        board = self.board if board is None else board
        map = np.zeros((self.size, self.size), dtype=np.int)
        for i in range(self.size):
            for j in range(self.size):
                if board[i, j]:
                    map[i, j] = self.anOtherPlayer(board[i, j])
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

        origin = np.array([[i + j for j in range(self.size)] for i in range(0, self.actionSize, self.size)])
        self.sameShapes = []
        self.sameShapes.extend(sequence(origin, flipX, flipY, flipX))
        self.sameShapes.extend(sequence(origin, rotate, flipX, flipY, flipX))
        self.sameShapes.extend(sequence(origin, rotate, rotate, flipX, flipY, flipX))
        self.sameShapes = distinct(self.sameShapes, lambda a, b: (a == b).all())

    def board2Str(self, board=None):
        board = self.board if board is None else board
        k, t = 0, 0
        res = []
        for i in range(self.size):
            for j in range(self.size):
                t = t * 3 + board[i][j]
                if k == 5:
                    res.append(chr(t))
                    k = 0
                    t = 0
                else:
                    k += 1
        if k > 0:
            res.append(chr(t))
        return ''.join(res)

    def anOtherPlayer(self, player):
        return 3 - player

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int)
        self.player = 2
        self._createMap()
        return copy.deepcopy(self.board)

    def step(self, action):
        x, y = action % self.size, action // self.size
        self.player = self.anOtherPlayer(self.player)
        self.board = self.getNextBoard(y * self.size + x)
        self._updateGrid(x, y)
        winner = self.getWinner()
        if winner is not None:
            winner = self.player if winner == 1 else 0
        return copy.deepcopy(self.board), self.player, winner

    def getNextBoard(self, action, board=None):
        board = self.board if board is None else board
        board = self.flip(board)
        board[action // self.size, action % self.size] = 1
        return board

    def extendInt(self, x):
        r = []
        for shape in self.sameShapes:
            r.append(shape[x // self.size][x % self.size])
        return r

    def extendList(self, l):
        r = []
        for shape in self.sameShapes:
            t = [0] * len(l)
            for i in range(len(l)):
                idx = shape[i // self.size][i % self.size]
                t[i] = l[idx]
            r.append(t)
        return r

    def extendBoard(self, b):
        r = []
        for shape in self.sameShapes:
            t = [[0] * self.size for i in range(self.size)]
            for i in range(self.size):
                for j in range(self.size):
                    idx = shape[i][j]
                    t[i][j] = b[idx // self.size][idx % self.size]
            r.append(t)
        return r

    def getWinner(self, board=None):
        board = self.board if board is None else board

        def checkLine():
            for i in range(self.size):
                for j in range(self.size - self.winLength + 1):
                    c = board[i][j]
                    if not c: continue
                    for k in range(1, self.winLength):
                        if board[i][j + k] != c:
                            break
                    else:
                        return c
            return None

        def checkColumn():
            for i in range(self.size - self.winLength + 1):
                for j in range(self.size):
                    c = board[i][j]
                    if not c: continue
                    for k in range(1, self.winLength):
                        if board[i + k][j] != c:
                            break
                    else:
                        return c
            return None

        def checkDiagonal():
            for i in range(self.size - self.winLength + 1):
                for j in range(self.size - self.winLength + 1):
                    c = board[i][j]
                    if not c: continue
                    for k in range(1, self.winLength):
                        if board[i + k][j + k] != c:
                            break
                    else:
                        return c
            for i in range(self.size - self.winLength + 1):
                for j in range(self.winLength - 1, self.size):
                    c = board[i][j]
                    if not c: continue
                    for k in range(1, self.winLength):
                        if board[i + k][j - k] != c:
                            break
                    else:
                        return c
            return None

        def checkFull():
            for i in range(self.size):
                for j in range(self.size):
                    if board[i][j] == 0:
                        return False
            return True

        winner = checkLine() or checkColumn() or checkDiagonal()
        if winner:
            return winner
        elif checkFull():
            return 0
        else:
            return None

    def _createMap(self):
        if not self.rendering:
            return
        if not hasattr(self, 'canvas'):
            width, height = self.MAZE_W * self.size, self.MAZE_H * self.size
            right, bottom = width, height
            self.canvas = tk.Canvas(self, bg='white', height=height, width=width)
            for c in range(0, right, self.MAZE_W):
                self.canvas.create_line(c, 0, c, bottom)
            for r in range(0, bottom, self.MAZE_H):
                self.canvas.create_line(0, r, right, r)
            self.canvas.pack()
            self.canvas.bind('<Button-1>', self.onClick)
        self.grids = []
        for i in range(self.size):
            self.grids.append([0] * self.size)
            for j in range(self.size):
                self._updateGrid(j, i)

    def _updateGrid(self, x, y):
        if not self.rendering:
            return
        p = self.board[y][x]
        if p != 0:
            p = p if self.player == 1 else self.anOtherPlayer(p)
        if self.grids[y][x]:
            self.canvas.delete(self.grids[y][x])
        self.grids[y][x] = self.canvas.create_rectangle(
            (x + 0.5) * self.MAZE_W - 0.5 * self.GRID_W, (y + 0.5) * self.MAZE_H - 0.5 * self.GRID_H,
            (x + 0.5) * self.MAZE_W + 0.5 * self.GRID_W, (y + 0.5) * self.MAZE_H + 0.5 * self.GRID_H,
            fill=self.TYPE2COLOR.get(p)
        )

    def render(self):
        if not self.rendering:
            return
        self.update()

    def onClick(self, event):
        x, y = event.x // self.MAZE_W, event.y // self.MAZE_H
        self.onGridClick and self.onGridClick(x, y)
