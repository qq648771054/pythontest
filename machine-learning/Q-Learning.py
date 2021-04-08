import numpy as np
import pandas as pd

EPSILON = 0.9 # 贪婪因子
ALPHA = 0.1 # 学习率
GAMMA = 0.9 # 衰减因子

ACTIONS = [[0, 1], [1, 0], [0, -1], [-1, 0]]
MAP = [
    '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', 'o', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', 'i', '.',
    '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
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
        a = np.random.randint(0, len(ACTIONS))
    else:
        idx = width * y + x
        a = q_table.iloc[idx, :].idxmax()
    tx, ty = x + ACTIONS[a][0], y + ACTIONS[a][1]
    if tx < 0 or ty < 0 or tx >= width or ty >= height:
        return genNextStep(map, q_table, width, height, x, y)
    else:
        if map[y][x] == 'i':
            return a, 100
        else:
            return a, 0

def run():
    width, height, x, y = analyzeMap(MAP)
    q_table = pd.DataFrame(np.zeros(width * height, len(ACTIONS)))
    while MAP[y][x] != 'i':
        a, r = genNextStep(MAP, q_table, width, height, x, y)
        q_predict = q_table.loc[y * width + x, a]
        x, y = x + ACTIONS[a][0], y + ACTIONS[a][1]
        q_target = r + GAMMA * q_table.iloc[y * width + x, :].max()
        

if __name__ == '__main__':
    print(run())