import numpy as np
arr = np.array([
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 2, 2, 0],
    [1, 1, 0, 0],
])


def board2Str(board, size):
    k = 0
    t = 0
    res = []
    for i in range(size):
        for j in range(size):
            t = t * 4 + board[i][j]
            if k == 3:
                res.append(chr(t))
                t = 0
                k = 0
            else:
                k += 1
    return ''.join(res)

print(board2Str(arr, 4))