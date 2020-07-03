import numpy as np

import torch

def get_absolute_yolo_box(y_pred, valid_anchors_wh, num_landmarks):
    box_xy, box_wh, objectness, landmarks = torch.split(y_pred, (2, 2, 1, num_landmarks), dim=-1)

    box_xy = torch.sigmoid(box_xy)
    box_wh = box_wh
    objectness = torch.sigmoid(objectness)
    landmarks_probs = torch.sigmoid(landmarks)
    bbox_rel = torch.cat((box_xy, box_wh), dim=-1)

    grid_size = y_pred.shape[1]
    grid_x, grid_y = torch.meshgrid(torch.arange(end=grid_size, dtype=torch.float, device=box_xy.device),
                                    torch.arange(end=grid_size, dtype=torch.float, device=box_xy.device))
    grid = torch.stack((grid_y, grid_x), dim=-1).unsqueeze_(2)

    box_xy = box_xy + grid
    box_xy = box_xy / float(grid_size)

    box_wh = torch.exp(box_wh) * valid_anchors_wh
    bbox_abs = torch.cat((box_xy.float(), box_wh.float()), dim=-1)

    return bbox_abs, objectness, landmarks_probs, bbox_rel

def get_relative_yolo_box(y_true, valid_anchors_wh, num_landmarks):
    box_xy, box_wh, objectness, landmarks = torch.split(y_true, (2, 2, 1, num_landmarks), dim=-1)

    bbox_abs = torch.cat((box_xy, box_wh), dim=-1)

    grid_size = y_true.shape[1]
    grid_x, grid_y = torch.meshgrid(torch.arange(grid_size, dtype=torch.float, device=box_xy.device),
                                    torch.arange(grid_size, dtype=torch.float, device=box_xy.device))
    grid = torch.stack((grid_y, grid_x), dim=-1).unsqueeze_(2)

    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    true_xy = true_xy * float(grid_size) - grid

    true_wh = torch.log(true_wh / valid_anchors_wh)
    true_wh = torch.where(
        torch.logical_or(torch.isinf(true_wh), torch.isnan(true_wh)),
        torch.zeros_like(true_wh), true_wh)

    bbox_rel = torch.cat((true_xy, true_wh), dim=-1)

    return bbox_rel, objectness, landmarks, bbox_abs

def broadcast_iou(box_1, box_2):
    box_1, box_2 = torch.broadcast_tensors(box_1, box_2)

    ze = torch.zeros(box_1[..., 2].shape).cuda()
    int_w = torch.max(torch.min(box_1[..., 2], box_2[..., 2]) -
                      torch.max(box_1[..., 0], box_2[..., 0]), ze)
    int_h = torch.max(torch.min(box_1[..., 3], box_2[..., 3]) -
                      torch.max(box_1[..., 1], box_2[..., 1]), ze)

    int_area = int_w * int_h

    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])

    return int_area / (box_1_area + box_2_area - int_area)


def xywh_to_x1x2y1y2(box):
    xy = box[..., 0:2]
    wh = box[..., 2:4]

    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2

    y_box = torch.cat([x1y1, x2y2], dim=-1)
    return y_box