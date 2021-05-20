import time

from Lib import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus= tf.config.list_physical_devices('GPU')
if len(gpus) > 0: tf.config.experimental.set_memory_growth(gpus[0], True)

def addAixs(arr):
    return arr.reshape((1, ) + arr.shape)

def copyModel(model):
    path = '_temp.h5'
    model.save(path)
    model = tf.keras.models.load_model(path)
    os.remove(path)
    return model
