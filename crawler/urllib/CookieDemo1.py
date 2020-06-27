# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 16:55
@file: CookieDemo1.py
@desc: 
"""
import http.cookiejar, urllib.request

cookie = http.cookiejar.CookieJar()
handler = urllib.request.HTTPCookieProcessor(cookie)
opener = urllib.request.build_opener(handler)
response = opener.open('http://www.baidu.com')
for item in cookie:
    print(item.name + "=:" + item.value)
