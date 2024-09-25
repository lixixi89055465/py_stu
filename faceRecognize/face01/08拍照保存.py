import cv2

# 读取视频
cap = cv2.VideoCapture(0)
flag = 1
num = 1

while (cap.isOpened()):
	ret_flag, Vshow = cap.read()
	cv2.imshow('Capture_Test', Vshow)
	k = cv2.waitKey(1) & 0xFF  # 按键判断
	if k == ord('s'):  # 保存
		cv2.imwrite('D:/workspace/py_stu/face_recognize/' + str(num) + '.name' + '.jpg', Vshow)
		print('success to save' + str(num) + '.jpg')
		print('-------------------------')
		num += 1
	elif k == ord(' '):  # 退出
		break
# 释放摄像头
cap.release()
# 是否内存
cv2.destroyAllWindows()
