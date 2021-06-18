from lib import *
import matplotlib.pyplot as plt
fileName = 'MountainCar_Continuous/MountainCar_Continuous_1/MountainCar_Continuous_1_32/log.txt'
def analyzeMaxHeight():
    file = open(getDataFilePath(fileName), 'r')
    lines = file.readlines()
    file.close()
    heights = []
    for line in lines:
        '2021-06-17 15:05:36.847446:agent 4, episode 1 step 999, totalStep 5526, max height 0.3082425895602159'
        '***************************sdsdsdsdsf'
        h = analyzeStr(line, '***************************sdsdsdsdsf')[-1]
        heights.append(h)
    hs = []
    step = 20
    for i in range(0, len(heights) - step, step):
        h = 0
        for j in range(step):
            h += heights[i + j]
        hs.append(h)
    episodes = [i + 1 for i in range(len(hs))]
    plt.scatter(episodes, hs)
    plt.show()

def analyzeStep():
    file = open(getDataFilePath(fileName), 'r')
    lines = file.readlines()
    file.close()
    steps = []
    for line in lines:
        '2021-06-17 15:05:36.847446:agent 4, episode 1 step 999, totalStep 5526, max height 0.3082425895602159'
        '***************************s     ds         ds     ds             ds               f'
        step = analyzeStr(line, '***************************s     ds         ds     ds             ds               f')[5]
        steps.append(step)
    ss = []
    step = 10
    for i in range(0, len(steps) - step, step):
        h = 0
        for j in range(step):
            h += steps[i + j]
        ss.append(h)
    episodes = [i + 1 for i in range(len(ss))]
    plt.scatter(episodes, ss)
    plt.show()

if __name__ == '__main__':
    # analyzeMaxHeight()
    analyzeStep()
