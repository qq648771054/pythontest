# -*- coding: utf-8 -*-
from lib import *
import os

class BQKan(object):
    '''笔趣看'''
    url = 'https://www.bqkan.com'

    @staticmethod
    def downloadText(url):
        class Visitor(DataVistor):
            def __init__(self):
                super(Visitor, self).__init__()
                self.texts = []

            def visit_Tag(self, data):
                if data.name == 'br':
                    self.texts.append('\n')

            def visit_NavigableString(self, data):
                self.texts.append(data.encode('utf-8'))

            def getText(self):
                return ''.join(self.texts)
        if not url.startswith('/'):
            url = '/' + url
        (datas, ) = getWebDatas(BQKan.url + url, '#content')
        visitor = Visitor()
        visitor.visit(datas)
        return visitor.getText()

    @staticmethod
    def downloadAllTexts(url, filepath=None, dirpath=None):
        class Visitor(DataVistor):
            def __init__(self):
                super(Visitor, self).__init__()
                self.texts = []
                self.tag = 0

            def visit_Tag(self, data):
                if data.name == 'dt':
                    self.tag += 1
                elif data.name == 'dd' and data.next.name == 'a':
                    if self.tag == 1:
                        data = data.next
                        href = data.get('href')
                        title = data.get_text().encode('utf-8')
                        self.texts.append({'href': href, 'title': title})

            def downloadInDir(self, dirpath):
                if not os.path.exists(dirpath):
                    os.mkdir(dirpath)
                for text in self.texts:
                    with open(os.path.join(dirpath, text['title'] + '.txt'), 'w') as f:
                        f.write(BQKan.downloadText(text['href']))

            def downloadInFile(self, filepath):
                pass

        (datas, ) = getWebDatas(url, 'body > div.listmain > dl', charset='gbk')
        visitor = Visitor()
        visitor.visit(datas)
        if dirpath:
            visitor.downloadInDir(dirpath)
        elif filepath:
            visitor.downloadInFile(filepath)
