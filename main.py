import random
import numpy as np
import math
import datetime
import sys
import msvcrt
import tensorflow as tf
import os

assert False

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpus= tf.config.list_physical_devices('GPU')
# if len(gpus) > 0: tf.config.experimental.set_memory_growth(gpus[0], True)
#
# def copyModel(model):
#     path = '_temp.h5'
#     model.save(path)
#     model = tf.keras.models.load_model(path)
#     os.remove(path)
#     return model
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, input_shape=(2, )),
#     tf.keras.layers.Dense(2, )
# ])
# model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MSE)
#
# model1 = copyModel(model)
# A = [1, 2]
# B = [2, 3]
# C = [7, 2]
# D = [1, 4]
# xs = [A, B]
# ys = [C, D]
# model.fit(np.array([A]), np.array([C]), epochs=2)
# model.fit(np.array([B]), np.array([D]), epochs=1)
# model1.fit(np.array([A, B]), np.array([C, D]), sample_weight=np.array([0, 0]), epochs=1)
#
#
# print(model.predict(xs))
# print(model1.predict(xs))
#
# print([1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:12])


