# -*- coding: utf-8 -*-

import time

def minInterval(interval):
    lastCall = [0]
    def f(func):
        def f1(*args, **kwargs):
            if lastCall[0] + interval >= time.time():
                time.sleep(lastCall[0] + interval - time.time())
            lastCall[0] = time.time()
            return func(*args, **kwargs)
        return f1
    return f


