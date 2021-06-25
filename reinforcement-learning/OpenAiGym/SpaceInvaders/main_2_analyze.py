from Lib import *
import matplotlib.pyplot as plt
fileName = 'SpaceInvaders/SpaceInvaders_2/SpaceInvaders_2_3/log.txt'

def showPltByStep(data, step, type='scatter'):
    img = []
    for i in range(0, len(data) - step, step):
        r = 0
        for j in range(step):
            r += data[i + j]
        img.append(r)
    episodes = [(i + 1) * step for i in range(len(img))]
    getattr(plt, type)(episodes, img)
    plt.show()

def analyzeRewards():
    file = open(getDataFilePath(fileName), 'r')
    lines = file.readlines()
    file.close()
    steps = []
    for line in lines:
        '2021-06-23 16:27:11.123239:agent: 2, episode: 1, step: 394, totalStep: 3554, rewards 50, actions [2, 1, 5, 5, 5, 2, 3, 4, 3, 4, 5, 4, 2, 4, 0, 3, 4, 3, 5, 2, 2, 0, 2, 0, 5, 5, 2, 5, 3, 5, 0, 4, 5, 0, 4, 0, 4, 0, 2, 2, 0, 3, 3, 0, 5, 4, 4, 3, 5, 5, 1, 0, 5, 2, 1, 5, 0, 2, 4, 1, 4, 5, 1, 4, 2, 0, 3, 0, 3, 3, 2, 0, 0, 1, 1, 2, 3, 4, 1, 5, 5, 5, 2, 4, 1, 3, 1, 4, 1, 3, 1, 4, 2, 5, 3, 4, 2, 2, 3, 0, 4, 5, 5, 5, 0, 5, 3, 4, 1, 1, 2, 5, 2, 1, 4, 2, 3, 4, 4, 3, 5, 0, 1, 5, 3, 3, 2, 4, 0, 5, 1, 2, 1, 4, 4, 0, 0, 5, 0, 3, 3, 4, 4, 2, 3, 0, 3, 4, 4, 5, 4, 0, 4, 4, 2, 2, 0, 0, 5, 2, 2, 5, 4, 0, 2, 2, 2, 4, 4, 1, 1, 5, 0, 1, 2, 5, 0, 3, 0, 0, 5, 3, 1, 0, 2, 0, 2, 2, 5, 1, 2, 5, 2, 2, 0, 4, 3, 3, 1, 0, 2, 2, 2, 1, 5, 3, 3, 3, 5, 1, 5, 0, 3, 3, 5, 5, 2, 5, 4, 0, 4, 4, 5, 5, 4, 0, 3, 2, 4, 2, 4, 5, 1, 4, 4, 3, 1, 4, 3, 2, 3, 1, 4, 4, 2, 2, 1, 0, 0, 2, 5, 5, 1, 3, 5, 3, 3, 4, 5, 2, 4, 5, 5, 4, 0, 2, 0, 3, 3, 3, 1, 5, 0, 2, 5, 3, 1, 1, 0, 2, 4, 4, 1, 2, 4, 2, 2, 1, 0, 5, 2, 1, 5, 0, 0, 1, 0, 3, 5, 0, 1, 0, 4, 5, 3, 1, 0, 3, 1, 5, 3, 1, 3, 0, 0, 2, 0, 0, 2, 1, 3, 5, 1, 5, 2, 0, 1, 1, 0, 1, 0, 0, 3, 4, 1, 1, 1, 0, 5, 4, 3, 0, 4, 5, 5, 3, 4, 0, 0, 1, 0, 3, 2, 5, 2, 5, 0, 0, 3, 2, 3, 0, 0, 5, 4, 2, 5, 5, 5, 0, 1, 4, 4, 5, 4, 2, 5, 4, 2, 2, 5, 1, 5, 4, 4, 2, 2, 0, 2, 0, 2, 0, 4, 3]'
        '***************************s      ds          ds       ds              ds            d **********a'
        res = analyzeStr(line,
                          '***************************s      ds          ds       ds              ds            d **********a')
        if res[0]:
            steps.append(res[0][9])

    showPltByStep(steps, 10, 'scatter')

if __name__ == '__main__':
    analyzeRewards()
