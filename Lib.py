# -*- coding: utf-8 -*-
import random
import numpy as np
import time
import datetime
import copy
import sys
import os
from collections import deque
import math
import cv2

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

def softmax(arr):
    return normallize([math.exp(a) for a in arr])

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

def clamp(arr, l, r):
    if hasattr(arr, '__iter__'):
        return [min(max(x, l), r) for x in arr]
    else:
        return min(max(arr, l), r)

def vstack(arrs):
    res = [[] for i in range(len(arrs[0]))]
    for arr in arrs:
        for i, a in enumerate(arr):
            res[i].append(a)
    return res

def hstack(arrs):
    res = []
    for arr in arrs:
        for a in arr:
            res.append(a)
    return res

def second2Str(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def min_to_zero(a, b):
    return a if abs(a) < abs(b) else b

def max_to_zero(a, b):
    return a if abs(a) > abs(b) else b

def analyzeStr(str, *formats):
    def isNum(c):
        return ord(c) >= ord('0') and ord(c) <= ord('9')

    def getFormat(str, ps, type, num):
        if type == 'd':
            n = 0
            assert isNum(str[ps]), 'not a number'
            while ps < len(str) and isNum(str[ps]):
                n = n * 10 + ord(str[ps]) - ord('0')
                ps += 1
            return int(n), ps
        elif type == 'f':
            n = 0
            assert isNum(str[ps]) or str[ps] == '.', 'not a number'
            while ps < len(str) and isNum(str[ps]):
                n = n * 10 + ord(str[ps]) - ord('0')
                ps += 1
            if str[ps] == '.':
                ps += 1
            scale = 0.1
            while ps < len(str) and isNum(str[ps]):
                n += scale * (ord(str[ps]) - ord('0'))
                scale *= 0.1
                ps += 1
            return float(n), ps
        elif type == 's':
            if num > 0:
                return str[ps: ps + num], ps + num
            else:
                pb = ps
                while ps < len(str) and not isNum(str[ps]):
                    ps += 1
                return str[pb: ps], ps
        elif type == '*':
            num = num if num > 0 else 1
            return None, ps + num
        elif type == ' ':
            return None, ps
        elif type == 'a':
            assert str[ps] == '[', '[ no find'
            pe = ps + 1
            while pe < len(str) and str[pe] != ']':
                pe += 1
            assert pe < len(str), '] no find'
            return eval(str[ps: pe + 1]), pe + 1
        else:
            assert False, 'unknown type ' + type

    ret = []
    for index, format in enumerate(formats):
        try:
            res = []
            ps = 0
            num = 0
            for i, c in enumerate(format):
                if isNum(c):
                    num = num * 10 + ord(c) - ord('0')
                else:
                    r, ps = getFormat(str, ps, c, num)
                    if r is not None:
                        res.append(r)
                    num = 0
            ret.append(res)
        except:
            ret.append(None)
    return ret

def rgb2Gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(img.shape + (1, ))
    return img