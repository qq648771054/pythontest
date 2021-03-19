# -*- coding: utf-8 -*-

import time
import sys
import os
from collections import deque

def minInterval(interval):
    '''最小调用间隔时间'''
    lastCall = [0]
    def f(func):
        def f1(*args, **kwargs):
            if lastCall[0] + interval >= time.time():
                time.sleep(lastCall[0] + interval - time.time())
            lastCall[0] = time.time()
            return func(*args, **kwargs)
        return f1
    return f

class pattern_input(object):
    '''格式化输入'''
    def __init__(self):
        self.input = self.readFromInput()
        self.input.next()
        self.buffers = deque()

    def pushStr(self, s):
        self.buffers.append({'type': str, 'val': s})

    def pushFile(self, f):
        self.buffers.append({'type': file, 'val': f})

    def empty(self):
        return len(self.buffers) == 0

    def genNewLine(self):
        while True:
            if len(self.buffers) > 0:
                s = self.buffers[0]
                if s['type'] == file:
                    f = open(s['val'], 'r')
                    while True:
                        line = f.readline()
                        if not line: break
                        yield line
                    f.close()
                else:
                    yield str(s['val'])
                self.buffers.popleft()
            else:
                yield None

    def readFromInput(self):
        yield 0
        newline = self.genNewLine()
        readOnly = False
        while True:
            s = newline.next()
            if s is None:
                if readOnly:
                    yield None
                    continue
                else:
                    s = raw_input()
            stt = 0
            for i, c in enumerate(s):
                if c in [' ', '\n', '\r', '\t']:
                    if i != stt:
                        readOnly = yield s[stt: i]
                    stt = i + 1
            if stt != len(s):
                readOnly = yield s[stt: len(s)]

    def readInput(self, T, count=None):
        return T(self.input.send(False)) if count is None else [T(self.input.send(False)) for i in xrange(count)]

    def read(self, T, count=None):
        if count is None:
            r = self.input.send(True)
            return r if r is None else T(r)
        else:
            r = [self.input.send(True) for i in xrange(count)]
            for i, t in enumerate(r):
                r[i] = r[i] if r[i] is None else T(r[i])
            return r

std_input = pattern_input()

def getDataFilePath(name):
    return os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data', name)
