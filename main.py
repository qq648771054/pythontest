import numpy as np
t = np.zeros((10, 10), dtype=np.int)
for i in range(10):
    for j in range(10):
        t[i, j] = i * 10 + j
print(t)