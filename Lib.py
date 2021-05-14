# -*- coding: utf-8 -*-
import random
import numpy as np
import time
import sys
import os
from collections import deque

def minInterval(interval):
    """最小调用间隔时间"""
    lastCall = [0]
    def f(func):
        def f1(*args, **kwargs):
            if lastCall[0] + interval >= time.time():
                time.sleep(lastCall[0] + interval - time.time())
            lastCall[0] = time.time()
            return func(*args, **kwargs)
        return f1
    return f

def calculateTime(func):
    """打印函数锁花费的时间"""
    def f(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        print('run method {} takes {} seconds'.format(func.__name__, time.time() - start))
        return ret
    return f

class pattern_input(object):
    """格式化输入"""
    def __init__(self):
        self.input = self.readFromInput()
        self.buffers = deque()
        self.break_in = False

    def pushStr(self, s):
        self.buffers.append({'type': 'str', 'val': s})

    def pushFile(self, f):
        self.buffers.append({'type': 'file', 'val': f})

    def pushStdIn(self):
        self.buffers.append({'type': 'in'})

    def breakStdIn(self):
        self.break_in = True

    def empty(self):
        return len(self.buffers) == 0

    def genNewLine(self):
        while True:
            if len(self.buffers) > 0:
                s = self.buffers.popleft()
                if s['type'] == 'file':
                    f = open(s['val'], 'r', encoding='utf-8')
                    while True:
                        line = f.readline()
                        if not line: break
                        yield line
                    f.close()
                elif s['type'] == 'str':
                    yield str(s['val'])
                elif s['type'] == 'in':
                    while not self.break_in:
                        yield sys.stdin.readline()
                    self.break_in = False
            else:
                yield None

    def readFromInput(self):
        newline = self.genNewLine()
        while True:
            s = next(newline)
            if s is None:
                yield None
                continue
            stt = 0
            for i, c in enumerate(s):
                if c in [' ', '\n', '\r', '\t']:
                    if i != stt:
                        yield s[stt: i]
                    stt = i + 1
            if stt != len(s):
                yield s[stt: len(s)]

    def get(self, T, count=None):
        return T(next(self.input)) if count is None else [T(next(self.input)) for i in range(count)]

    def generateData(self, *args, limit=None):
        data = [[] for i in range(len(args))]
        try:
            lines = 0
            while limit == None or lines < limit:
                for i, arg in enumerate(args):
                    if isinstance(arg, (list, tuple)):
                        data[i].append([self.get(a) for a in arg])
                    else:
                        data[i].append(self.get(arg))
                lines += 1
        finally:
            return data

std_input = pattern_input()

def getDataFilePath(name):
    """返回数据的绝对路径"""
    return os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data', name)

def readFile(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        texts = f.read()
    return texts

def countArr(arr):
    """给数组元素计数"""
    r = {}
    for a in arr:
        r[a] = r.get(a, 0) + 1
    return r

def transform(arr, func=None, dict=None):
    """用规则将数组转换成新数组"""
    if hasattr(arr, '__iter__'):
        if func:
            return [func(a) for a in arr]
        elif dict:
            return [dict[a] for a in arr]
    else:
        if func:
            return func(arr)
        elif dict:
            return dict[arr]

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

def distinct(arr, eq=lambda a, b: a == b):
    res = []
    for a in arr:
        for r in res:
            if eq(a, r):
                break
        else:
            res.append(a)
    return res


def sample(arr, size):
    if len(arr) <= size:
        return [a for a in arr]
    else:
        idx = np.random.choice(len(arr), size=size)
        return [arr[i] for i in idx]
