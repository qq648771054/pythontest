import numpy as np
arr = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
])


def rotate(board):
    return np.transpose(board)[::-1]


def flipX(board):
    return board[:, ::-1]


def flipY(board):
    return board[::-1, :]

print(arr)
print((flipX(flipX(arr)) == arr).all())
print(flipY(arr))
# print(rotate(arr))
# print(rotate(rotate(arr)))
# print(rotate(rotate(rotate(arr))))