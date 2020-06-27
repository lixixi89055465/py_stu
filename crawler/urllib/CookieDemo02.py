# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 17:02
@file: CookieDemo02.py
@desc: 
"""
import http.cookiejar, urllib.request

filename = 'cookie.txt'
cookie = http.cookiejar.MozillaCookieJar(filename)
handler = urllib.request.HTTPCookieProcessor(cookie)
opener = urllib.request.build_opener(handler)
response = opener.open('http://www.baidu.com')
for item in cookie:
    print(item.name + "=:" + item.value)
