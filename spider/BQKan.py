# -*- coding: utf-8 -*-
from lib import *
import os

class BQKan():
    name = '笔趣看'
    url = 'https://www.bqkan.com'

def getUrl(url):
    if url.startswith('http://') or url.startswith('https://'):
        return url
    elif not url.startswith('/'):
        return BQKan.url + '/' + url
    else:
        return BQKan.url + url

class Visitor_downloadText(DataVistor):
    def __init__(self):
        super(Visitor_downloadText, self).__init__()
        self.texts = []

    def visit_Tag(self, data):
        if data.name == 'br':
            self.texts.append('\n')

    def visit_NavigableString(self, data):
        self.texts.append(data.encode('utf-8'))

    def getText(self):
        return ''.join(self.texts)

def downloadText(url):
    (datas,) = getWebDatas(getUrl(url), '#content')
    visitor = Visitor_downloadText()
    visitor.visit(datas)
    return visitor.getText()


class Visitor_downloadAllTexts(DataVistor):
    def __init__(self):
        super(Visitor_downloadAllTexts, self).__init__()
        self.texts = []
        self.tag = 0

    def visit_Tag(self, data):
        if data.name == 'dt':
            self.tag += 1
        elif data.name == 'dd' and data.next.name == 'a':
            if self.tag == 2:
                data = data.next
                href = data.get('href')
                title = data.get_text()
                self.texts.append({'href': href, 'title': title})

    def downloadInDir(self, dirpath):
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        print 'download to {}:\n'.format(dirpath)
        for i, text in enumerate(self.texts):
            print '{}/{}:{}'.format(i + 1, len(self.texts), text['title'].encode('utf-8'))
            with open(os.path.join(dirpath, text['title'] + '.txt'), 'w') as f:
                f.write(downloadText(text['href']))

def downloadAllTexts(url, dirpath):
    (datas, ) = getWebDatas(getUrl(url), 'body > div.listmain > dl')
    visitor = Visitor_downloadAllTexts()
    visitor.visit(datas)
    visitor.downloadInDir(dirpath)

if __name__ == '__main__':
    downloadAllTexts('0_790/', dirpath=r'C:\Users\wb.zhangshenglong\Desktop\test')
    print downloadText('0_790/36873824.html')