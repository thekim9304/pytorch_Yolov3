import cv2

import torch
import torch.nn as nn

from yolov3 import YoloV3, anchors_wh
from postprocess import Postprocessor

if __name__=='__main__':
    num_landmarks = 136

    input = torch.randn(2, 3, 416, 416).cuda()

    img = cv2.imread('E:/DB_FaceLandmark/300VW_frame/train/001_0000.jpg')
    img = cv2.resize(img, (416, 416))
    x = torch.from_numpy(img / 255).float().unsqueeze(0).permute(0, -1, 1, 2).cuda()

    model = YoloV3(num_landmarks=num_landmarks)
    # model = YoloV3(num_landmarks=num_landmarks, backbone_network='darknet53')
    state = torch.load('C:/Users/th_k9/Desktop/pytorch_Yolov3/model/340_0.0022.pth')
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    model.cuda()

    y_pred = model(x, training=False)

    medium = y_pred[1]

    # print('bbox_abs', medium[0])
    # print('objectness', medium[1])
    # print('landmarks_probs', medium[2])
    # print('box_rel', medium[3])

    postprocessor = Postprocessor(max_detection=3, iou_threshold=0.5, score_threshold=0.5).cuda()

    boxes, scores, landmarks, num_detection = postprocessor(y_pred)
    #
    # landmarks = landmarks * img.shape[0]
    #
    # num_img = num_detection.shape[0]
    # for img_i in range(num_img):
    #     # based on train data
    #     # h, w, d = x[img_i].shape
    #     # based on original image
    #     h, w, d = img.shape
    #
    #     for i in range(num_detection.item()):
    #         box = boxes.cpu()[img_i][i]
    #
    #         xmin = int(box[0] * w)
    #         ymin = int(box[1] * h)
    #         xmax = int(box[2] * w)
    #         ymax = int(box[3] * h)
    #
    #         img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    #
    #     for i in range(0, 136, 2):
    #         img = cv2.circle(img, (int(landmarks[0][img_i][i]), int(landmarks[0][img_i][i+1])), 2, (0, 0, 255), -1)
    #
    # cv2.imshow('t', img)
    # # cv2.imwrite('C:/Users/th_k9/Desktop/fl_model_test.jpg', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()