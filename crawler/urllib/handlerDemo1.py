# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 16:45
@file: handlerDemo1.py
@desc: 
"""
import urllib.request
proxy_handler=urllib.request.ProxyHandler({
    'http':'http://127.0.0.1:9743',
    'https': 'http://127.0.0.1:9743',
})
opener=urllib.request.build_opener(proxy_handler)
response=opener.open('http://localhost/get')
print(response.read())