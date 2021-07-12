import cv2 as cv
src=cv.imread('./data/demo.png')
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
cv.waitKey(0)
cv.destroyAllWindows()

print('Hi,python!')