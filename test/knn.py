# -*- coding: utf-*-
from Lib import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kNN


def knn(inx, data, type, k, use01=True):
    if use01:
        maxData = np.max(data, axis=0)
        minData = np.min(data, axis=0)
        data = (data - minData) / (maxData - minData)
        inx = (inx - minData) / (maxData - minData)
    diff = np.sum((inx - data) ** 2, axis=1) ** 0.5
    arr = [type[i] for i in diff.argsort()[:k]]
    r = countArr(arr)
    return max(r)

def test_knn(data, type, k, ratio=None, test_data=None, test_type=None,
             useLib=False, **kwargs):
    if ratio != None:
        testCount = int(len(data) * ratio)
        test_data = data[: testCount]
        test_type = type[: testCount]
        data = data[testCount:]
        type = type[testCount:]
    else:
        testCount = len(test_data)
    correct = 0
    if useLib:
        neigh = kNN(n_neighbors=k, algorithm='auto')
        neigh.fit(data, type)
        eq = np.equal(neigh.predict(test_data), test_type)
        for i in eq:
            correct += 1 if i else 0
    else:
        for d, t in zip(test_data, test_type):
            if knn(d, data, type, k, **kwargs) == t:
                correct += 1
    return correct / float(testCount)

@calculateTime
def classic_movie(inx, k):
    std_input.pushFile(getDataFilePath('knn-classic_movie.txt'))
    data, type = std_input.generateData((float, float), str)
    data = np.array(data)
    return knn(inx, data, type, k)

@calculateTime
def classic_helen(inx, k):
    std_input.pushFile(getDataFilePath('knn-Helen.txt'))
    titles = std_input.get(str, count=3)
    data, type = std_input.generateData((float, float, float), str)
    data = np.array(data)
    return knn(inx, data, type, k)

def show_helen():
    std_input.pushFile(getDataFilePath('knn-Helen.txt'))
    titles = std_input.get(str, count=3)
    data, type = std_input.generateData((float, float, float), str)
    data = np.array(data)
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))
    map = {'didntLike': 'blue', 'smallDoses': 'orange', 'largeDoses': 'red'}
    colors = transform(type, lambda x: map[x])
    def draw(axes, a, b):
        axes.scatter(data[:, a], data[:, b], color=colors)
        axes.set_title(u'{}与{}的关系'.format(titles[a].decode('utf-8'), titles[b].decode('utf-8')))
        axes.set_xlabel(titles[a].decode('utf-8'))
        axes.set_ylabel(titles[b].decode('utf-8'))
        import matplotlib.lines as mlines
        didntLike = mlines.Line2D([], [], color='blue', marker='.', markersize=6, label='didntLike')
        smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
        largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
        axes.legend(handles=[didntLike, smallDoses, largeDoses])

    draw(axes[0, 0], 0, 1)
    draw(axes[0, 1], 0, 2)
    draw(axes[1, 0], 1, 2)
    figure.show()

@calculateTime
def test_helen():
    std_input.pushFile(getDataFilePath('knn-Helen.txt'))
    titles = std_input.get(str, count=3)
    data, type = std_input.generateData((float, float, float), str)
    data = np.array(data)
    return test_knn(data, type, 3, ratio=0.3)

def generateData_digit(path):
    data, type = [], []
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                content = f.read()
                arr = []
                for d in content:
                    if d in ['0', '1']:
                        arr.append(int(d))
                data.append(arr)
                type.append(int(file[0]))
    return np.array(data), type

@calculateTime
def test_digit():
    data, type = generateData_digit(getDataFilePath('knn-trainingDigits'))
    testData, testType = generateData_digit(getDataFilePath('knn-testDigits'))
    t1 = time.time()
    r1 = test_knn(data, type, 3, test_data=testData, test_type=testType, use01=False)
    t1 = time.time() - t1
    t2 = time.time()
    r2 = test_knn(data, type, 3, test_data=testData, test_type=testType, useLib=True)
    t2 = time.time() - t2
    print 'use self:\npredict {}, time:{}\nuse lib:\npredict {}:, time:{}'.format(r1, t1, r2, t2)

if __name__ == '__main__':
    # print classic_movie(np.array([101, 20]), 3)
    # print classic_helen(np.array([1, 1.0, 2.0]), 6)
    # show_helen()
    # print test_helen()
    test_digit()
