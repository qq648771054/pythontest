# -*- coding: utf-*-
import numpy as np
from numpy.random.mtrand import randn
import matplotlib.pyplot as plt
if __name__ == '__main__':
    points = np.arange(-5, 5, 0.01)
    xs, ys = np.meshgrid(points, points)
    z = np.sqrt(xs ** 2 + ys ** 2)
    plt.title(r'image plot of $\sqrt{x^2 + y ^2}$ for a  grid of values')
    plt.imshow(z, cmap=plt.cm.gray)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()