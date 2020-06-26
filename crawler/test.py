# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/26 21:26
@file: test.py
@desc: 
"""
from selenium import webdriver

driver=webdriver.Chrome()
driver.get('http://www.baidu.com')
driver.page_source