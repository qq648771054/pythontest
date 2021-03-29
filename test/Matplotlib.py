import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(np.arange(1, 5), np.arange(2, 6), np.arange(3, 7))
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    fig.show()