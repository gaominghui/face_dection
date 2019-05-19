# coding=utf-8

import os
import cv2
import dlib

cameraCapture = cv2.VideoCapture(0)
success, frame = cameraCapture.read()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")
win = dlib.image_window()
win.clear_overlay()
while success and cv2.waitKey(1) == -1:
    success, frame = cameraCapture.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 调整画面尺寸
    win.set_image(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)              #生成灰度图
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #生成直方图
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1)
    win.clear_overlay()
    for k, d in enumerate(detections): 
        shape = predictor(clahe_image, d)  # 获取坐标
        for i in range(1, 68):  # 每张脸都有68个识别点
             cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255),
                       thickness=2)
        win.add_overlay(shape)

        # 绘制矩阵轮廓
        win.add_overlay(detections[k])

    #cv2.imshow("Camera", frame)

cameraCapture.release()
cv2.destroyAllWindows()
