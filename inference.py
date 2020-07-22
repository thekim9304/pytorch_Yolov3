import cv2
import time
import numpy as np

import torch
import torch.nn as nn

from yolov3 import YoloV3, anchors_wh
from postprocess import Postprocessor

def infer_one_frame():
    num_landmarks = 96

    input = torch.randn(2, 3, 416, 416).cuda()

    img = cv2.imread('E:/DB_FaceLandmark/300VW_frame/train/001_0000.jpg')
    # img = cv2.imread('E:/DB_FaceLandmark/300W/01_Indoor/indoor_048.png')
    img = cv2.resize(img, (416, 416))
    x = torch.from_numpy(img / 255).float().unsqueeze(0).permute(0, -1, 1, 2).cuda()

    model = YoloV3(num_landmarks=num_landmarks, backbone_network='darknet53')
    # state = torch.load('C:/Users/th_k9/Desktop/pytorch_Yolov3/model/ubuntu_606_0.0308.pth')
    state = torch.load('E:/models/309_0.0099.pth')

    model.load_state_dict(state['model_state_dict'])
    model.eval()
    model.cuda()

    y_pred = model(x, training=False)

    postprocessor = Postprocessor(max_detection=3, iou_threshold=0.5, score_threshold=0.5).cuda()

    boxes, scores, landmarks, num_detection = postprocessor(y_pred)

    landmarks = landmarks * img.shape[0]

    num_img = num_detection.shape[0]
    for img_i in range(num_img):
        # based on train data
        # h, w, d = x[img_i].shape
        # based on original image
        h, w, d = img.shape

        for i in range(num_detection.item()):
            box = boxes.cpu()[img_i][i]

            xmin = int(box[0] * w)
            ymin = int(box[1] * h)
            xmax = int(box[2] * w)
            ymax = int(box[3] * h)

            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        for i in range(0, 96, 2):
            img = cv2.circle(img, (int(landmarks[0][img_i][i]), int(landmarks[0][img_i][i+1])), 2, (0, 0, 255), -1)

    cv2.imshow('t', img)
    # cv2.imwrite('C:/Users/th_k9/Desktop/fl_model_test.jpg', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def infer_cam():
    num_landmarks = 96

    model = YoloV3(num_landmarks=num_landmarks, backbone_network='darknet53')
    # state = torch.load('C:/Users/th_k9/Desktop/pytorch_Yolov3/model/ubuntu_606_0.0308.pth')
    state = torch.load('E:/models/309_0.0099.pth')
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    model.cuda()

    # cap = cv2.VideoCapture(0)
    # 113, 143
    cap = cv2.VideoCapture('E:/DB_FaceLandmark/300VW/007/vid.avi')

    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('C:/Users/th_k9/Desktop/143.avi', fourcc, 30.0, (416, 416))

    while True:
        ret, frame = cap.read()

        if ret:
            img = cv2.resize(frame, (416, 416))
            x = torch.from_numpy(img / 255).float().unsqueeze(0).permute(0, -1, 1, 2).cuda()

            y_pred = model(x, training=False)

            postprocessor = Postprocessor(max_detection=3, iou_threshold=0.5, score_threshold=0.5).cuda()

            boxes, scores, landmarks, num_detection = postprocessor(y_pred)

            landmarks = landmarks * img.shape[0]

            num_img = num_detection.shape[0]
            for img_i in range(num_img):
                # based on train data
                # h, w, d = x[img_i].shape
                # based on original image
                h, w, d = img.shape

                for i in range(num_detection.item()):
                    box = boxes.cpu()[img_i][i]

                    xmin = int(box[0] * w)
                    ymin = int(box[1] * h)
                    xmax = int(box[2] * w)
                    ymax = int(box[3] * h)

                    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                for i in range(0, 96, 2):
                    img = cv2.circle(img, (int(landmarks[0][img_i][i]), int(landmarks[0][img_i][i+1])), 1, (255, 255, 255), -1)

            cv2.imshow('t', img)
            # out.write(img)

            if cv2.waitKey(1) == 27:
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
    # out.release()


if __name__=='__main__':
    # infer_one_frame()
    infer_cam()