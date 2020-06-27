# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 17:15
@file: CookieDemo04.py
@desc: 
"""
import http.cookiejar,urllib.request
filename='cookie4.txt'
cookie=http.cookiejar.MozillaCookieJar(filename)
handler=urllib.request.HTTPCookieProcessor(cookie)
opener=urllib.request.build_opener(handler)
response=opener.open('http://www.baidu.com')
cookie.save(ignore_discard=True,ignore_expires=True)

