import numpy as np
import pandas as pd
import time

EPSILON = 0.9 # 贪婪因子
ALPHA = 0.1 # 学习率
GAMMA = 0.99 # 衰减因子

ACTIONS = [[0, 1], [1, 0], [0, -1], [-1, 0]]
MAP = [
    ['.', '.', '*', '.', '.', '.', '.', '.', '.', '.'],
    ['.', 'o', '*', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '*', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '*', '*', '.', '*', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['*', '*', '*', '*', '.', '*', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '*', '*', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '*', 'i', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '*', '*', '.'],
    ]

def analyzeMap(map):
    height = len(map)
    width = len(map[0])
    si, sj, = 0, 0
    for i in range(height):
        for j in range(width):
            if map[i][j] == 'o':
                si, sj = i, j
                map[i][j] = '.'
    return height, width, si, sj

def genNextStep(map, q_table, width, height, x, y):
    if np.random.uniform() >= EPSILON:
        return np.random.randint(0, len(ACTIONS))
    else:
        idx = width * y + x
        return q_table.iloc[idx, :].idxmax()

def getNextState(map, q_table, width, height, x, y, a):
    tx, ty = x + ACTIONS[a][0], y + ACTIONS[a][1]
    if tx < 0 or ty < 0 or tx >= width or ty >= height or map[ty][tx] == '*':
        return x, y, -100
    else:
        if map[ty][tx] == 'i':
            return tx, ty, 1000
        else:
            return tx, ty, -1

def showMap(map, width, height, x, y):
    s = ''
    for i in range(height):
        for j in range(width):
            if i == y and j == x:
                s += 'o'
            else:
                s += map[i][j]
        s += '\n'
    print(s)

def run():
    width, height, sy, sx = analyzeMap(MAP)
    q_table = pd.DataFrame(np.zeros([width * height, len(ACTIONS)]))
    epoch = 0
    while True:
        step = []
        x, y = sy, sx
        epoch += 1
        while MAP[y][x] != 'i':
            a = genNextStep(MAP, q_table, width, height, x, y)
            tx, ty, r = getNextState(MAP, q_table, width, height, x, y, a)
            q_predict = q_table.loc[y * width + x, a]
            q_target = r + GAMMA * q_table.iloc[ty * width + tx, :].max()
            q_table.loc[y * width + x, a] += ALPHA * (q_target - q_predict)
            x, y = tx, ty
            # showMap(MAP, width, height, x, y)
            step.append([x, y])
            # time.sleep(1)
        print('epoch {} :use {} steps to arrive\n{}'.format(epoch, len(step), step))

if __name__ == '__main__':
    print(run())