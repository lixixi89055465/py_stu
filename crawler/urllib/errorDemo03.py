# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 17:49
@file: errorDemo03.py
@desc: 
"""
import socket
import urllib.request
import urllib.error

try:
    response = urllib.request.urlopen('https://www.baidu.com', timeout=0.01)
except urllib.error.URLError as e:
    print(type(e.reason))
    if isinstance(e.reason, socket.timeout):
        print('TIME OUT')
