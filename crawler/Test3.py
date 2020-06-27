# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 13:14
@file: Test3.py
@desc: 
"""
from selenium import webdriver
# 启动谷歌浏览器
driver=webdriver.Chrome()
# 进入指定网址
# driver.get('https://sina.cn/index/feed?from=touch&Ver=20')
driver.get('https://www.zhihu.com')

