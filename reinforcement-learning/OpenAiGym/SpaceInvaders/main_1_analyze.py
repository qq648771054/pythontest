from Lib import *
import matplotlib.pyplot as plt
fileName = 'SpaceInvaders/SpaceInvaders_1/SpaceInvaders_1_4/log.txt'

def analyzeRewards():
    file = open(getDataFilePath(fileName), 'r')
    lines = file.readlines()
    file.close()
    i = 0
    steps = []
    for line in lines:
        i += 1
        '2021-06-22 19:06:22.171325:random play, memory size 5284 step: 399, rewards: 60'
        '***************************6s'
        '2021-06-22 19:18:34.233595:episode: 29, step: 687, totalStep: 23588, rewards: 105'
        '***************************s        ds        ds              ds              d'
        res = analyzeStr(line,
                          '***************************6s',
                          '***************************s          ds         ds              ds              d')
        if res[1]:
            steps.append(res[1][3])
    r = []
    step = 1
    for i in range(0, len(steps) - step, step):
        h = 0
        for j in range(step):
            h += steps[i + j]
        r.append(h)
    episodes = [i + 1 for i in range(len(r))]
    plt.scatter(episodes, r)
    plt.show()

if __name__ == '__main__':
    analyzeRewards()
