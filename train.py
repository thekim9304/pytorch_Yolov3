import os
import time
import datetime

import torch
from torch.utils.data import DataLoader

from yolov3 import YoloV3, YoloLoss
from yolov3 import anchors_wh_mask, anchors_wh
from preprocess import CustomDataset
from postprocess import Postprocessor
from utils import get_absolute_yolo_box

BATCH_SIZE = 8
EPOCH = 10000


def main():
    ann_dir = 'C:/Users/th_k9/Desktop/Yolov3withFacelandmark/annotation_preparation/300VW_train2'
    # ann_dir = 'E:/DB_FaceLandmark/300VW+WFLW'
    img_dir = 'E:/DB_FaceLandmark'
    pth_dir = 'E:/models/'

    saved_pth_dir = 'C:/Users/th_k9/Desktop/pytorch_Yolov3/model'
    pth_file = 'darknet_309_0.0099.pth'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_landmarks = 96
    lr_rate = 0.00007
    pre_train = True

    dataset = CustomDataset(ann_dir, img_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # postprocessor = Postprocessor(max_detection=3, iou_threshold=0.5, score_threshold=0.5).cuda()

    model = YoloV3(num_landmarks, 'darknet53').to(device)
    loss_objects = [YoloLoss(valid_anchors_wh, num_landmarks) for valid_anchors_wh in anchors_wh_mask]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    model_save_cnt = 0

    if pre_train:
        state = torch.load(os.path.join(saved_pth_dir, pth_file))
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        init_epoch = state['Epoch']
        lowest_loss = state['loss']
    else:
        init_epoch = 0
        lowest_loss = 0.01

    print('init_epoch : {}'.format(init_epoch))
    print('lowest_loss : {}'.format(lowest_loss))

    for epoch in range(init_epoch, EPOCH, 1):
        model.train()
        print('{} epoch start! : {}'.format(epoch, datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")))

        epoch_total_loss = train_one_epoch(model, loss_objects, dataloader, optimizer, use_cuda)

        if epoch_total_loss < lowest_loss:
            lowest_loss = epoch_total_loss
            state = {
                'Epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_total_loss
            }
            file_path = pth_dir + '{}_{:.4f}.pth'.format(epoch, lowest_loss)
            torch.save(state, file_path)
            print('Save model _ [loss : {:.4f}, save_path : {}]\n'.format(lowest_loss, file_path))
            # if model_save_cnt == 0:
            #     lowest_loss = epoch_total_loss
            #     state = {
            #         'Epoch' : epoch,
            #         'model_state_dict' : model.state_dict(),
            #         'optimizer_state_dict' : optimizer.state_dict(),
            #         'loss' : epoch_total_loss
            #     }
            #     file_path = pth_dir + '{}_{:.4f}.pth'.format(epoch, lowest_loss)
            #     torch.save(state, file_path)
            #     print('Save model _ [loss : {:.4f}, save_path : {}]\n'.format(lowest_loss, file_path))
            #     model_save_cnt += 1
            # elif model_save_cnt == 9:
            #     print('model_save_cnt reset!')
            #     model_save_cnt = 0
            # else:
            #     lowest_loss = epoch_total_loss
            #     print('epoch_total_loss < lowest_loss, but model_save_cnt is {}'.format(model_save_cnt))
            #     model_save_cnt += 1

        if lowest_loss < 0.0001:
            print('lowest_loss too low : {}'.format(lowest_loss))
            break

def train_one_epoch(model, loss_objects, dataloader, optimizer, use_cuda, post=None):
    len_dataloader = len(dataloader)
    epoch_total_loss = 0.0
    epoch_xy_loss = 0.0
    epoch_wh_loss = 0.0
    epoch_landmark_loss = 0.0
    epoch_obj_loss = 0.0

    for i, data in enumerate(dataloader):
        x, label = data

        if use_cuda:
            x = x.cuda()
            label[0] = label[0].cuda()
            label[1] = label[1].cuda()
            label[2] = label[2].cuda()

        optimizer.zero_grad()

        model_output = model(x, training=True)

        '''
        '''
        # box_small = get_absolute_yolo_box(model_output[0], anchors_wh[0:3], 136)
        # box_medium = get_absolute_yolo_box(model_output[1], anchors_wh[3:6], 136)
        # box_large = get_absolute_yolo_box(model_output[2], anchors_wh[6:9], 136)

        # print(label[1][0][8][16][0][4])
        # print(box_medium[1][0][8][16][0])
        # print(box_medium[1][0][4][16][0])

        # post((box_small, box_medium, box_large))
        '''
        '''

        total_losses, xy_losses, wh_losses, landmark_losses, obj_losses = [], [], [], [], []
        for loss_object, y_pred, y_true in zip(loss_objects, model_output, label):
            total_loss, loss_breakdown = loss_object(y_true, y_pred)
            xy_loss, wh_loss, landmark_loss, obj_loss = loss_breakdown
            total_losses.append(total_loss * (1. / BATCH_SIZE))
            xy_losses.append(xy_loss * (1. / BATCH_SIZE))
            wh_losses.append(wh_loss * (1. / BATCH_SIZE))
            landmark_losses.append(landmark_loss * (1. / BATCH_SIZE))
            obj_losses.append(obj_loss * (1. / BATCH_SIZE))

        total_losses = torch.sum(torch.stack(total_losses))
        total_xy_loss = torch.sum(torch.stack(xy_losses))
        total_wh_loss = torch.sum(torch.stack(wh_losses))
        total_landmark_loss = torch.sum(torch.stack(landmark_losses))
        total_obj_loss = torch.sum(torch.stack(obj_losses))

        total_losses.backward()
        optimizer.step()

        epoch_total_loss += ((total_losses.item()) / len_dataloader)
        epoch_xy_loss += ((total_xy_loss.item()) / len_dataloader)
        epoch_wh_loss += ((total_wh_loss.item()) / len_dataloader)
        epoch_landmark_loss += ((total_landmark_loss.item()) / len_dataloader)
        epoch_obj_loss += ((total_obj_loss.item()) / len_dataloader)

    print(' total_loss.[{0:.4f}]'.format(epoch_total_loss))
    print('     xy:{:.4f}, wh:{:.4f}, landmark:{:.4f}, obj:{:.4f}'.format(epoch_xy_loss,
                                                                          epoch_wh_loss,
                                                                          epoch_landmark_loss,
                                                                          epoch_obj_loss))

    return epoch_total_loss

if __name__ == '__main__':
    main()
