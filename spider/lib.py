# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup

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