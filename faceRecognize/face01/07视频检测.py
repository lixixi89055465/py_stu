import cv2 as cv


def face_detect_demo(img):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	face_detect = cv.CascadeClassifier(
		'C:/anaconda3/envs/py39/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
	face = face_detect.detectMultiScale(gray)
	for x, y, w, h in face:
		cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
	cv.imshow('result', img)


# 读取视频
cap = cv.VideoCapture('1.mp4')

while True:
	flag, frame = cap.read()
	if not flag:
		break
	face_detect_demo(frame)
	if ord('q') == cv.waitKey(0):
		break
# 释放内存
cv.destroyAllWindows()
# 释放摄像头
cap.release()
cv.destroyAllWindows()
