# -*- coding: utf-8 -*-
# @Time : 2024/9/13 18:10
# @Author : nanji
# @Site : 
# @File : 01读取图片.py
# @Software: PyCharm 
# @Comment :
import cv2 as cv

img = cv.imread('face1.jpg')
cv.imshow('read_img', img)
cv.waitKey(0)

cv.destroyAllWindows()
