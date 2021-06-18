# -*- coding: utf-8 -*-
from Lib import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# gpus= tf.config.list_physical_devices('GPU')
# if len(gpus) > 0: tf.config.experimental.set_memory_growth(gpus[0], True)

def addAixs(arr):
    return arr.reshape((1, ) + arr.shape)

def copyModel(model0, model1=None):
    if model1:
        model0.set_weights(model1.get_weights())
    else:
        path = '_temp.h5'
        model0.save(path)
        model0 = tf.keras.models.load_model(path)
        os.remove(path)
        return model0

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



