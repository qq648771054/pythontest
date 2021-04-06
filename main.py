# -*- coding: utf-*-
from Lib import *
import pytz
import time
import datetime
nowTime = datetime.datetime.fromtimestamp(time.time())
cntz = pytz.timezone('Asia/Shanghai')
tz = pytz.timezone('America/Los_Angeles')
currentTime = tz.localize(nowTime).astimezone(cntz)
print(currentTime, nowTime.astimezone(cntz))
print(currentTime.timestamp(), nowTime.timestamp())
'''
16 17 18 19 20 21 22 23 00 01 02 03 04 05 06
20 21 22 23 00 01 02 03 04 05 06 07 08 09 10
'''
