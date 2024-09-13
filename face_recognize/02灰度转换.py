import cv2 as cv

img = cv.imread('face1.jpg')

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('gray', gray_img)

cv.imwrite('gray_fac1.jpg', gray_img)


cv.imshow('read_img', img)
cv.waitKey(0)

cv.destroyAllWindows()
