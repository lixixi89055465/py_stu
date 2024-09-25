import cv2 as cv

img = cv.imread('face1.jpg')

resize_img = cv.resize(img, dsize=(200, 200))
cv.imshow('img', img)
cv.imshow('resize_img', resize_img)
print('未修改:', img.shape)
print('修改后:', resize_img.shape)

cv.waitKey(0)
cv.destroyAllWindows()
