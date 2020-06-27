# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 15:37
@file: example5.py
@desc: 
"""
import socket
import urllib.request
import urllib.error

try:
    response=urllib.request.urlopen('http://localhost/get', timeout=0.1)
    print(response)
except urllib.error.URLError as e:
    if isinstance(e.reason, socket.timeout):
        print('TIME OUT')
