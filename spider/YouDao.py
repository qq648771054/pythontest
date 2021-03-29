# -*- coding: utf-8 -*-
from Lib import *
import random
import hashlib
import json
import traceback
import requests

class YouDao():
    name = '有道翻译'
    url = 'http://fanyi.youdao.com'

@minInterval(0.1)
def translate(word, fromLan='AUTO', toLan='AUTO'):
    url = r'http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule'
    ts = str(int(time.time() * 1000))
    salt = ts + str(int(10 * random.random()))
    sign = hashlib.md5("fanyideskweb{}{}Tbh5E8=q6U3EXe+&L[4c@".format(word, salt)).hexdigest()
    formData = {
        'i': word,
        'from': fromLan,
        'to': toLan,
        'smartresult': 'dict',
        'client': 'fanyideskweb',
        'salt': salt,
        'sign': sign,
        'lts': ts,
        'bv': '3d91b10fc349bc3307882f133fbc312a',
        'doctype': 'json',
        'version': '2.1',
        'keyfrom': 'fanyi.web',
        'action': 'FY_BY_REALTlME'
    }
    headers = {
        'Cookie': 'OUTFOX_SEARCH_USER_ID=573768917@10.108.160.18; OUTFOX_SEARCH_USER_ID_NCOO=2027052426.385918; '
                  '_ga=GA1.2.222716885.1606202437; _ntes_nnid=87f8bc315c3b27f616fac110779599c6,1606467522047; '
                  'JSESSIONID=aaaaQxBvghuyE_uD_W9Cx; ___rl__test__cookies=1611653110468',
        'Referer': 'http://fanyi.youdao.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/88.0.4324.104 Safari/537.36 '
    }
    try:
        response = requests.post(url, data=formData, headers=headers)
        content = json.loads(response.text)
        return content['translateResult'][0][0]['tgt']
    except:
        print(traceback.print_exc())
        return None

if __name__ == '__main__':
    print(translate(raw_input('翻译内容:'), fromLan='zh-CHS', toLan='ja'))