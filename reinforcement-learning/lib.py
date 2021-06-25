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



