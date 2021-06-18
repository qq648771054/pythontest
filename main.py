import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math
import tensorflow as tf
import numpy as np
import time
import threading

def analyzeStr(str, format):
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
            assert num > 0, 'num must > 0'
            s = ps[ps: ps + num]
            return s, ps + num
        elif type == '*':
            num = num if num > 0 else 1
            return None, ps + num
        elif type == ' ':
            return None, ps
        else:
            assert False, 'unknown type ' + type

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
    return res





print(
analyzeStr('2021-06-17 15:05:36.847446:agent 4, episode 1 step 999, totalStep 5526, max height 0.3082425895602159',
           '*********************************d**********d  ****d  ************d     ***********f                 ')
)
