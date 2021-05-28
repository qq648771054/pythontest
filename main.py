import random
import numpy as np
import math


def validActions(state, player=None):
    SIZE = 5
    WIN_LENGTH = 4

    direct = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]

    def extend(x, y, dx, dy):
        tx, ty = x + dx, y + dy
        c = state[ty][tx] if 0 <= tx < SIZE and 0 <= ty < SIZE else 0
        if c != 0:
            r = 1
            while True:
                tx, ty = tx + dx, ty + dy
                if 0 <= tx < SIZE and 0 <= ty < SIZE and state[ty][tx] == c:
                    r += 1
                else:
                    break
            return r, c
        else:
            return 0, 0

    all = []
    m1, m2 = set(), set()
    for i in range(SIZE):
        for j in range(SIZE):
            if state[i][j] == 0:
                idx = i * SIZE + j
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
                    if l >= WIN_LENGTH - 1:
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

print(validActions(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 2, 0],
        [2, 1, 2, 0, 0],
        [0, 2, 0, 0, 0],
    ]
    , player=1))