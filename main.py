# -*- coding: utf-8 -*-

from spider import YouDao
from spider import BQKan

if __name__ == '__main__':
    pass
    # BQKan.downloadAllTexts('0_790/', dirpath=r'C:\Users\wb.zhangshenglong\Desktop\test')
    # print BQKan.downloadText('0_790/36873824.html')
    print YouDao.translate(raw_input('翻译内容:'), fromLan='zh-CHS', toLan='ja')