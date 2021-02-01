# -*- coding: utf-8 -*-

from lib import *
import web
import translation

#body > div.listmain > dl > nth-child()
if __name__ == '__main__':
    pass
    # web.BQKan.downloadAllTexts('0_790/', dirpath=r'C:\Users\wb.zhangshenglong\Desktop\test')
    # print web.BQKan.downloadText('0_790/36873824.html')
    print translation.YouDao.translate(raw_input('翻译内容:'), fromLan='zh-CHS', toLan='ja')

