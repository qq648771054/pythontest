# -*- coding: utf-*-
from Lib import *
class myTime(object):
    @staticmethod
    def time():
        return 123

sys.modules['time'] = myTime
import time
print(time.time())
