# coding=utf-8

from __future__ import print_function

import cv2
import dlib

cameraCapture = cv2.VideoCapture(0)
success, frame = cameraCapture.read()
detector = dlib.get_frontal_face_detector()

while success and cv2.waitKey(1) == -1:
    success, frame = cameraCapture.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 调整画面尺寸

    faces = detector(frame, 1)
    for k, d in enumerate(faces):
        frame = cv2.rectangle(frame, (d.left(), d.top()),
                              (d.right(), d.bottom()), (255, 0, 0), 2)

    cv2.imshow("Camera", frame)

cameraCapture.release()
cv2.destroyAllWindows()
