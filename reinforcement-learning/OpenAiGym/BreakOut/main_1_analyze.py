from Lib import *
import matplotlib.pyplot as plt
fileName = 'BreakOut/BreakOut_1/BreakOut_1_3/log.txt'

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
        '2021-06-25 09:46:09.972834 : agent: 5, episode: 193, step: 209, totalStep: 49741, rewards 0, actions [3, 0, 0, 0, 2, 3, 0, 2, 0, 2, 0, 0, 2, 3, 3, 3, 0, 2, 2, 2, 2, 0, 3, 2, 0, 0, 0, 3, 3, 3, 2, 3, 1, 2, 2, 3, 0, 0, 0, 3, 3, 2, 1, 0, 2, 2, 3, 2, 3, 3, 0, 2, 2, 0, 1, 3, 0, 2, 0, 2, 0, 1, 0, 2, 0, 3, 0, 2, 0, 0, 2, 0, 1, 0, 3, 1, 0, 3, 2, 0, 2, 2, 0, 3, 3, 0, 1, 1, 3, 0, 0, 3, 3, 0, 2, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 0, 0, 3, 2, 1, 3, 3, 2, 1, 0, 0, 2, 2, 0, 0, 2, 1, 3, 2, 2, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 3, 0, 0, 3, 3, 2, 3, 0, 1, 3, 3, 0, 1, 0, 0, 0, 3, 3, 1, 2, 3, 0, 0, 1, 3, 3, 2, 0, 2, 2, 0, 3, 0, 0, 3, 0, 0, 3, 3, 1, 3, 2, 0, 3, 3, 3, 3, 3, 2, 3, 0, 2, 2, 2, 3, 0, 0, 1, 2, 2, 0, 0, 0, 3, 3, 0, 3, 0, 2, 1, 3]'
        '***************************s        ds          ds         ds              ds             d**********a'
        res = analyzeStr(line,
                          '***************************s        ds          ds         ds              ds             d**********a')
        if res[0]:
            steps.append(res[0][9])

    showPltByStep(steps, 10, 'scatter')

if __name__ == '__main__':
    analyzeRewards()
