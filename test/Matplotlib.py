import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time

def data_gen(t=0):
    cnt = 0
    while cnt < 1000:
        cnt += 1
        t += 0.1
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)


def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []


def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
                              repeat=False, init_func=init)
plt.ion()
plt.show()

# if __name__ == '__main__':
#     xdata, ydata = [], []
#     fig, axes = plt.subplots()
#     ln, = axes.plot([], [], 'ro', animated=True)
#     def update(frame):
#         frame += 1
#         ln.set_data(np.arange(0, frame), np.arange(frame, 0, -1))
#         return ln,
#
#     def init():
#         axes.set_xlim(0, 100)
#         axes.set_ylim(0, 100)
#         return ln,
#
#     animation = animation.FuncAnimation(fig, update, interval=10, init_func=init)
#
#     while True:
#         animation._draw_frame()
#
#     plt.show()
