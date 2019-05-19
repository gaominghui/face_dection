# coding=utf-8
from __future__ import print_function

import cv2

cameraCapture = cv2.VideoCapture("blast2.avi")
success, frame = cameraCapture.read()

while success and cv2.waitKey(1) == -1:
    success, frame = cameraCapture.read()

    #TODO:在此处可放置各种对当前每一帧图像的处理
    cv2.imshow("Camera", frame)

cameraCapture.release()
cv2.destroyAllWindows()
