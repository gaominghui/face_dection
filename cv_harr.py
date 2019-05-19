# coding=utf-8

from __future__ import print_function

import cv2


def mark_face(img, x, y, w, h):
    return cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cameraCapture = cv2.VideoCapture(0)
success, frame = cameraCapture.read()

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')


while success and cv2.waitKey(1) == -1:
    success, frame = cameraCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    [mark_face(frame, *args) for args in faces]
    cv2.imshow("Camera", frame)

cameraCapture.release()
cv2.destroyAllWindows()
