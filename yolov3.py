import time
import numpy as np

from mobilnetv2 import MobileNetV2
from darknet53 import DarkConv, Darknet53
from utils import (get_absolute_yolo_box,
                   get_relative_yolo_box,
                   xywh_to_x1x2y1y2,
                   broadcast_iou)
from preprocess import CustomDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

anchors_wh = torch.tensor([[10, 13], [16, 30], [33, 23],
                           [30, 61], [62, 45], [59, 119],
                           [116, 90], [156, 198], [373, 326]]).float().cuda() / 416
anchors_wh_mask = torch.tensor([[[10, 13], [16, 30], [33, 23]],
                                [[30, 61], [62, 45], [59, 119]],
                                [[116, 90], [156, 198], [373, 326]]]).float().cuda() / 416


class YoloV3(nn.Module):
    def __init__(self, num_landmarks=196, backbone_network='mobilev2'):
        super(YoloV3, self).__init__()
        self.num_landmarks = num_landmarks

        final_filters = 3 * (4 + 1 + num_landmarks)

        if backbone_network is 'mobilev2':
            self.backbone = MobileNetV2()
            out_channels = [320, 96, 32]
        elif backbone_network is 'darknet53':
            self.backbone = Darknet53(in_ch=3)
            out_channels = [1024, 512, 256]
        else:
            out_channels = []

        self.large_scale_dbl5 = nn.Sequential(
            DarkConv(out_channels[0], 512, kernel_size=1, strides=1),
            DarkConv(512, 1024, kernel_size=3, strides=1),
            DarkConv(1024, 512, kernel_size=1, strides=1),
            DarkConv(512, 1024, kernel_size=3, strides=1),
            DarkConv(1024, 512, kernel_size=1, strides=1))
        self.large_scale_detection = nn.Sequential(
            DarkConv(512, 1024, kernel_size=3, strides=1),
            nn.Conv2d(1024, final_filters, kernel_size=1, stride=1, padding=0))

        self.large_scale_upsampling = nn.Sequential(
            DarkConv(512, 256, kernel_size=1, strides=1),
            nn.Upsample(scale_factor=2))
        self.medium_scale_block = nn.Sequential(
            DarkConv(out_channels[1] + 256, 256, kernel_size=1, strides=1),
            DarkConv(256, 512, kernel_size=3, strides=1),
            DarkConv(512, 256, kernel_size=1, strides=1),
            DarkConv(256, 512, kernel_size=3, strides=1),
            DarkConv(512, 256, kernel_size=1, strides=1))
        self.medium_scale_detection = nn.Sequential(
            DarkConv(256, 512, kernel_size=3, strides=1),
            nn.Conv2d(512, final_filters, kernel_size=1, stride=1, padding=0))

        self.medium_scale_upsampling = nn.Sequential(
            DarkConv(256, 128, kernel_size=1, strides=1),
            nn.Upsample(scale_factor=2))
        self.small_scale_block = nn.Sequential(
            DarkConv(out_channels[2] + 128, 128, kernel_size=1, strides=1),
            DarkConv(128, 256, kernel_size=3, strides=1),
            DarkConv(256, 128, kernel_size=1, strides=1),
            DarkConv(128, 256, kernel_size=3, strides=1),
            DarkConv(256, 128, kernel_size=1, strides=1))
        self.small_scale_detection = nn.Sequential(
            DarkConv(128, 256, kernel_size=3, strides=1),
            nn.Conv2d(256, final_filters, kernel_size=1, stride=1, padding=0))

    def forward(self, inputs, training):
        x_small, x_medium, x_large = self.backbone(inputs)

        x = self.large_scale_dbl5(x_large)
        y_large = self.large_scale_detection(x)

        x = self.large_scale_upsampling(x)
        x = torch.cat((x, x_medium), 1)
        x = self.medium_scale_block(x)
        y_medium = self.medium_scale_detection(x)

        x = self.medium_scale_upsampling(x)
        x = torch.cat((x, x_small), 1)
        x = self.small_scale_block(x)
        y_small = self.small_scale_detection(x)

        y_large_shape = y_large.shape
        y_medium_shape = y_medium.shape
        y_small_shape = y_small.shape

        y_large = y_large.view(y_large_shape[0], -1, 3, y_large_shape[-2], y_large_shape[-1]).permute(0, -1, -2, 2, 1)
        y_medium = y_medium.view(y_medium_shape[0], -1, 3, y_medium_shape[-2], y_medium_shape[-1]).permute(0, -1, -2, 2, 1)
        y_small = y_small.view(y_small_shape[0], -1, 3, y_small_shape[-2], y_small_shape[-1]).permute(0, -1, -2, 2, 1)

        if training:
            return y_small, y_medium, y_large

        box_small = get_absolute_yolo_box(y_small, anchors_wh[0:3], self.num_landmarks)
        box_medium = get_absolute_yolo_box(y_medium, anchors_wh[3:6], self.num_landmarks)
        box_large = get_absolute_yolo_box(y_large, anchors_wh[6:9], self.num_landmarks)
        return box_small, box_medium, box_large


class YoloLoss(nn.Module):
    def __init__(self, valid_anchors_wh, num_landmarks, ignore_thresh=0.5, lambda_coord=5.0, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.valid_anchors_wh = valid_anchors_wh
        self.num_landmarks = num_landmarks
        self.ignore_thresh = ignore_thresh
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, y_true, y_pred):
        """
            - y_pred to bbox_abs
            - get pred_xy_rel and pred_wh_rel
        """
        pred_box_abs, pred_obj, pred_landmark, pred_box_rel = get_absolute_yolo_box(y_pred,
                                                                                    self.valid_anchors_wh,
                                                                                    self.num_landmarks)
        # print(self.valid_anchors_wh)
        # print('=' * 20, 'get_abs_pred', '*' * 20)
        # print(pred_box_abs.shape)
        # print(pred_obj.shape)
        # print(pred_landmark.shape)
        # print(pred_box_rel.shape)
        pred_box_abs = xywh_to_x1x2y1y2(pred_box_abs)
        pred_xy_rel = pred_box_rel[..., 0:2]
        pred_wh_rel = pred_box_rel[..., 2:4]

        """
            - y_true to bbox_rel
            - get true_xy_rel and true_wh_rel
        """
        true_box_rel, true_obj, true_landmark, true_box_abs = get_relative_yolo_box(y_true,
                                                                                    self.valid_anchors_wh,
                                                                                    self.num_landmarks)

        # print(true_obj)
        # print('=' * 20, 'get_rel_pred', '*' * 20)
        # print(true_box_rel.shape)
        # print(true_obj.shape)
        # print(true_landmark.shape)
        # print(true_box_abs.shape)
        true_box_abs = xywh_to_x1x2y1y2(true_box_abs)
        true_xy_rel = true_box_rel[..., 0:2]
        true_wh_rel = true_box_rel[..., 2:4]

        true_wh_abs = true_box_abs[..., 2:4]
        weight = 2 - true_wh_abs[..., 0] * true_wh_abs[..., 1]

        # print('=' * 20, 'calc_loss', '*' * 20)
        xy_loss = self.calc_xy_loss(true_xy_rel, pred_xy_rel, true_obj, weight)
        wh_loss = self.calc_xy_loss(true_wh_rel, pred_wh_rel, true_obj, weight)
        landmark_loss = self.calc_xy_loss(true_landmark, pred_landmark, true_obj, weight)
        ignore_mask = self.calc_ignore_mask(true_box_abs, pred_box_abs, true_obj)

        # print('=' * 10, 'xy_loss', '=' * 10)
        # print(xy_loss)
        # print('=' * 10, 'wh_loss', '=' * 10)
        # print(wh_loss)
        # print('=' * 10, 'landmark_loss', '=' * 10)
        # print(landmark_loss)
        # print('-' * 10, 'obj_loss', '-' * 10)
        obj_loss = self.calc_obj_loss(true_obj, pred_obj, ignore_mask)

        return xy_loss + wh_loss + landmark_loss + obj_loss, (
        xy_loss, wh_loss, landmark_loss, obj_loss)

    def calc_xy_loss(self, true, pred, true_obj, weight):
        loss = torch.sum(torch.square(true - pred), dim=-1)
        true_obj = torch.squeeze(true_obj, dim=-1)
        loss = loss * true_obj * weight
        loss = torch.sum(loss, dim=(1, 2, 3)) * self.lambda_coord

        return loss

    def calc_ignore_mask(self, true_box, pred_box, true_obj):
        obj_mask = torch.squeeze(true_obj, dim=-1)

        best_iou = []
        for x in zip(pred_box, true_box, obj_mask):
            mask = x[1][x[2].bool()]
            if mask.size(0) is not 0:
                best_iou.append(broadcast_iou(x[0], mask))
            else:
                best_iou.append(torch.zeros(true_box.shape[1:4]).cuda())
        best_iou = torch.stack(best_iou)

        ignore_mask = (best_iou < self.ignore_thresh).float()
        ignore_mask = ignore_mask.unsqueeze(-1)

        return ignore_mask

    def calc_obj_loss(self, true_obj, pred_obj, ignore_mask):
        obj_entropy = self.binary_cross_entropy(pred_obj, true_obj)
        obj_loss = true_obj * obj_entropy
        noobj_loss = (1 - true_obj) * obj_entropy * ignore_mask

        obj_loss = torch.sum(obj_loss, dim=(1, 2, 3, 4))
        noobj_loss = torch.sum(noobj_loss, dim=(1, 2, 3, 4)) * self.lambda_noobj

        return obj_loss + noobj_loss

    def binary_cross_entropy(self, logits, labels):
        epsilon = 1e-7
        logits = torch.clamp(logits, epsilon, 1 - epsilon)

        return -(labels * torch.log(logits) +
                 (1 - labels) * torch.log(1 - logits))


if __name__ == '__main__':
    # lambda_coord = 5.0
    # lambda_noobj = 0.5
    # ignore_thresh = 0.5

    dataset = CustomDataset('C:/Users/th_k9/Desktop/Yolov3withFacelandmark/annotation_preparation/300VW_train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    input = torch.randn(2, 3, 416, 416).cuda()

    model = YoloV3(136).cuda()

    loss_func1 = YoloLoss(anchors_wh_mask[0], 136).cuda()
    loss_func2 = YoloLoss(anchors_wh_mask[1], 136).cuda()
    loss_func3 = YoloLoss(anchors_wh_mask[2], 136).cuda()

    for batch, data in enumerate(dataloader):
        image, label = data

        y_pred = model(image.cuda(), training=True)

        # loss1, loss_breakdown1 = loss_func1(label[0], y_pred[0])
        loss2, loss_breakdown2 = loss_func2(label[1], y_pred[1])
        # loss3, loss_breakdown3 = loss_func3(label[2], y_pred[2])



        if batch == 0:
            break
