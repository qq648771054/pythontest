from Lib import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def addAixs(arr):
    return arr.reshape((1, ) + arr.shape)

def copyModel(model):
    path = '_temp.h5'
    model.save(path)
    model = tf.keras.models.load_model(path)
    os.remove(path)
    return model
