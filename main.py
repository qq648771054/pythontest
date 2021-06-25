
import math
import numpy as np
import time
import threading
import os
import random
from distutils import sysconfig
import matplotlib.pyplot as plt
import tensorflow as tf

# 降低tensorflow警告等级
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 配置GPU内存
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

def copyModel(model0, model1=None):
    if model1:
        model0.set_weights(model1.get_weights())
    else:
        path = '_temp.h5'
        model0.save(path)
        model0 = tf.keras.models.load_model(path)
        os.remove(path)
        return model0

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

m1 = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2, )),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
m1.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy
)
m2 = copyModel(m1)
m3 = copyModel(m1)
size = 100
x = [normallize([i, i + 1]) for i in range(size)]
y = [normallize([i * 2, i - 1]) for i in range(size)]
m2.fit(np.array(x), np.array(y), sample_weight=np.array([0.5] * size), epochs=1, verbose=0)
m3.fit(np.array(x), np.array(y) * 0.5, epochs=1, verbose=0)
print(m1.predict(np.array(x)))
print(m2.predict(np.array(x)))
