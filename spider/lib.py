# -*- coding: utf-8 -*-

import time
import requests
from bs4 import BeautifulSoup

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

def getWebDatas(url, *elements):
    strhtml = requests.get(url)
    strhtml.encoding = strhtml.apparent_encoding
    soup = BeautifulSoup(strhtml.content, 'lxml')
    res = []
    for ele in elements:
        res.append(soup.select(ele))
    return res

class DataVistor(object):
    def visit(self, datas):
        for item in datas:
            for c in item.contents:
                visitor = 'visit_' + c.__class__.__name__
                if hasattr(self, visitor):
                    getattr(self, visitor)(c)
