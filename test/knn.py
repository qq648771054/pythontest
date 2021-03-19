# -*- coding: utf-*-
from Lib import *
import numpy as np

def classic_move(cmp, k):
    std_input.pushFile(getDataFilePath('knn-classic_movie.txt'))
    data = []
    type = []
    while True:
        a, l, t = std_input.read(int), std_input.read(int), std_input.read(str)
        if t is not None:
            data.append([a, l])
            type.append(t)
        else:
            break
    data = np.array(data, dtype=int)
    dataSize = data.shape[0]
    diff = np.tile(cmp, (dataSize, 1))
    print diff

if __name__ == '__main__':
    classic_move(np.array([3, 51]), 2)
