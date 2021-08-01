import cv2 as cv
import numpy as  np


def fill_color_demo(image):
    copyImage = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv.floodFill(copyImage, mask, (30, 30), (0, 255, 255), (100, 100, 100), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill_color_demo", copyImage)


src = cv.imread('./data/demo.jpg')
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
fill_color_demo(src)
'''
 print(src.shape)
 face = src[400:950, 400:900]
 
 cv.imshow("face", face)
 gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
 cv.imshow("gray", gray)
 
 backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
 cv.imshow("backface", backface)
 
 src[400:950, 400:900] = backface
 cv.imshow("inline backface", src)
'''

cv.waitKey(0)
cv.destroyAllWindows()
print('Hi,python!')
