import random
import numpy as np
import math

def normallize(arr):
    total = 0
    for a in arr:
        total += a
    if total > 0.0:
        scale = 1.0 / total
        n = []
        for a in arr:
            n.append(a * scale)
        return n
    else:
        return [1.0 / len(arr)] * len(arr)

def softmax(arr):
    return normallize([math.exp(a) for a in arr])

print(softmax([1, 2, 3, 4, 5]))