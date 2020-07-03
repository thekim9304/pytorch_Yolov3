import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

anchors_wh = torch.tensor([[10, 13], [16, 30], [33, 23],
                       [30, 61], [62, 45], [59, 119],
                       [116, 90], [156, 198], [373, 326]]).float().cuda() / 416



DB_PATH = 'C:/Users/th_k9/Desktop/Yolov3withFacelandmark/annotation_preparation/300VW_train'

class CustomDataset(Dataset):
    def __init__(self, ann_dir, img_dir, output_shape=(416, 416), transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.output_shape = output_shape
        self.anno_paths = []

        files = os.listdir(ann_dir)
        if 'annotations' in files:
            ann_dir = os.path.join(ann_dir, 'annotations')
        else:
            raise Exception('No exist the \'annotations\' file')

        self.all_anns = self.parse_annotation(ann_dir)

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.all_anns)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.all_anns[idx]['path'])
        image = cv2.resize(cv2.imread(img_path), self.output_shape) / 255
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        bbox, landmark = self.parse_y_feature(self.all_anns[idx])
        label = (
            self.preprocess_label_for_one_scale(bbox, landmark,
                                                52, torch.tensor([0, 1, 2]).cuda()),
            self.preprocess_label_for_one_scale(bbox, landmark,
                                                26, torch.tensor([3, 4, 5]).cuda()),
            self.preprocess_label_for_one_scale(bbox, landmark,
                                                13, torch.tensor([6, 7, 8]).cuda())
        )
        return image, label

    def preprocess_label_for_one_scale(self, bboxes, landmarks, grid_size, valid_anchors):
        y = torch.zeros((grid_size, grid_size, 3, 5 + landmarks.shape[-1])).cuda()
        anchor_indices = self.find_best_anchors(bboxes)

        num_boxes = landmarks.shape[0]

        for i in range(num_boxes):
            curr_landmarks = torch.tensor(landmarks[i]).cuda().float()
            curr_box = torch.tensor(bboxes[i]).cuda().float()
            curr_anchor = anchor_indices[i]

            if curr_anchor in valid_anchors:
                adjusted_anchor_index = curr_anchor % 3

                # yolo loss 계산하기 위해
                # (xmin, ymin, xmax, ymax)를 (centroid x, centroid y, width, height)로 변환
                curr_box_xy = (curr_box[..., 0:2] + curr_box[..., 2:4]) / 2
                curr_box_wh = curr_box[..., 2:4] - curr_box[..., 0:2]

                grid_cell_xy = curr_box_xy // float(1 / grid_size)

                index = torch.tensor([grid_cell_xy[1], grid_cell_xy[0], adjusted_anchor_index.item()]).int().cuda()
                update = torch.cat((curr_box_xy, curr_box_wh, torch.tensor([1.0]).float().cuda(), curr_landmarks),
                                   dim=0)

                y[index[0]][index[1]][index[2]] = update

        return y

    def find_best_anchors(self, bboxes):
        box_wh = torch.from_numpy(bboxes[..., 2:4] - bboxes[..., 0:2]).float().cuda()
        box_wh = box_wh.unsqueeze(1).repeat(1, anchors_wh.shape[0], 1).float()

        intersection = torch.min(box_wh[..., 0], anchors_wh[..., 0]) * torch.min(box_wh[..., 1], anchors_wh[..., 1])
        box_area = box_wh[..., 0] * box_wh[..., 1]
        anchor_area = anchors_wh[..., 0] * anchors_wh[..., 1]

        iou = intersection / (box_area + anchor_area - intersection)

        return torch.argmax(iou, dim=-1).int()

    def parse_y_feature(self, anns):
        bboxes = []
        landmarks = []

        for obj in anns['object']:
            bboxes.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            landmarks.append(obj['landmarks'])

        return np.array(bboxes), np.array(landmarks)

    def parse_annotation(self, ann_dir):
        all_anns = []

        for ann in sorted(os.listdir(ann_dir)):
            img = {'object': []}

            tree = ET.parse(os.path.join(ann_dir, ann))

            for elem in tree.iter():
                if 'path' in elem.tag:
                    img['path'] = elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag:
                    obj = {}

                    for attr in list(elem):
                        if 'bbox' in attr.tag:
                            img['object'] += [obj]
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text))) / img['width']
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text))) / img['height']
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text))) / img['width']
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text))) / img['height']

                        if 'landmarks' in attr.tag:
                            obj['landmarks'] = np.array(list((map(float, attr.text.split(' ')))))
                            idx = np.arange(obj['landmarks'].shape[0])
                            obj['landmarks'][idx[0::2]] = obj['landmarks'][idx[0::2]] / img['width']
                            obj['landmarks'][idx[1::2]] = obj['landmarks'][idx[1::2]] / img['height']


            if len(img['object']) > 0:
                all_anns.append(img)

        return all_anns


def show_landmarks(image, landmarks):
    image = image.permute(1, 2, 0).numpy()

    """Show image with landmarks"""
    landmarks = np.array([landmarks[i * 2:(i + 1) * 2] for i in range((len(landmarks) + 2 - 1) // 2)])
    plt.imshow(image)
    plt.scatter(landmarks[:, 0]*image.shape[1], landmarks[:, 1]*image.shape[0], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__=='__main__':
    dataset = CustomDataset(DB_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    image, label = dataset[2]

    #
    #
    # plt.figure()
    # show_landmarks(image, label[1][7][16][2][5:].tolist())
    # plt.show()
