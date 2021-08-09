import cv2 as cv
import numpy as np


def add_demo(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow("add_demo", dst)


def subtract_demo(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", dst)


def divide_demo(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow("divide_demo", dst)


def multiply_demo(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow("multiply_demo", dst)


def other(m1, m2):
    # dst1 = cv.mean(m1)
    # dst2 = cv.mean(m2)
    M1, dev1 = cv.meanStdDev(m1)
    M2, dev2 = cv.meanStdDev(m2)

    print(M1)
    print(M2)
    print(dev1)
    print(dev2)
    h, w = m1.shape[:2]
    img = np.zeros([h, w], np.uint8)
    m, dev = cv.meanStdDev(img)
    print(m)
    print(dev)


def logic_demo(m1, m2):
    # dst=cv.bitwise_and(m1,m2)
    # dst=cv.bitwise_or(m1,m2)
    dst = cv.bitwise_not(m1, m2)
    cv.imshow("bitwise_and", dst)

def contrast_brightness_demo(image,c,b):
    h,w,ch=image.shape
    blank=np.zeros([h,w,ch],image.dtype)
    dst=cv.addWeighted(image,c,blank,1-c,b)
    cv.imshow("con-bri-demo",dst)



print("======= Hello Python =======")
src1 = cv.imread("./images/LinuxLogo.jpg")
src2 = cv.imread("./images/WindowsLogo.jpg")
src3 = cv.imread("./images/demo.jpg")

cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# print(src1.shape)
# print(src2.shape)
# cv.imshow("image1", cv.bitwise_not(src3))
# cv.imshow("image2", src2)
src=cv.imread("./images/demo.png")
cv.imshow("image2",src)
contrast_brightness_demo(src,1.2,100)
add_demo(src1, src2)
# subtract_demo(src1, src2)
# divide_demo(src1, src2)
# multiply_demo(src1, src2)
# logic_demo(src1, src2)
# other(src1, src2)
cv.waitKey(0)
cv.destroyAllWindows()
