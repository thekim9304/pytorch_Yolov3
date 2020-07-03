import torch
import torch.nn as nn

from utils import xywh_to_x1x2y1y2, broadcast_iou


class Postprocessor(nn.Module):
    def __init__(self, iou_threshold=0.5, score_threshold=0.5, max_detection=100):
        super(Postprocessor, self).__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detection = max_detection

    def forward(self, raw_yolo_outputs):
        boxes, objectness, landmarks_coord = [], [], []

        for raw_yolo_out in raw_yolo_outputs:
            # raw_yolo_out : (bbox_abs, objectness, landmarks_probs, bbox_rel)
            # print(raw_yolo_out[1].shape)
            batch_size = raw_yolo_out[0].size(0)
            num_landmarks = raw_yolo_out[2].size(-1)
            boxes.append(raw_yolo_out[0].view(batch_size, -1, 4))
            if raw_yolo_out[1].shape[1] == 26:
                pass
                # print(raw_yolo_out[1].contiguous().view(batch_size, -1, 1)[0][76])

                # val = 0
                # d_val = 0
                # for d, i in enumerate(raw_yolo_out[1].contiguous().view(batch_size, -1, 1)[0]):
                #     print(i)
                #     if i > val:
                #         val = i
                #         d_val = d
                # print(d_val, val)

            objectness.append(raw_yolo_out[1].contiguous().view(batch_size, -1, 1))
            landmarks_coord.append(raw_yolo_out[2].contiguous().view(batch_size, -1, num_landmarks))

        boxes = xywh_to_x1x2y1y2(torch.cat(boxes, dim=1))
        objectness = torch.cat(objectness, dim=1)
        landmark_coord = torch.cat(landmarks_coord, dim=1)

        return self.batch_non_maximum_suppression(boxes, objectness, landmark_coord)

    def batch_non_maximum_suppression(self, boxes, objectness, landmark_coord):
        def single_batch_nms(candidate_boxes):
            y_mask = candidate_boxes[..., 4] >= self.score_threshold
            candidate_boxes = candidate_boxes[y_mask]
            outputs = torch.zeros((self.max_detection + 1, candidate_boxes.size(-1)))

            indices = []
            updates = []

            count = 0
            while candidate_boxes.size(0) > 0 and count < self.max_detection:
                best_idx = torch.argmax(candidate_boxes[..., 4], dim=0)
                best_box = candidate_boxes[best_idx]

                indices.append([count] * candidate_boxes.size(-1))
                updates.append(best_box)
                count += 1

                candidate_boxes = torch.cat(
                    (candidate_boxes[0:best_idx], candidate_boxes[best_idx + 1:candidate_boxes.size(0)]), dim=0)

                iou = broadcast_iou(best_box[0:4], candidate_boxes[..., 0:4])

                iou_mask = iou <= self.iou_threshold
                candidate_boxes = candidate_boxes[iou_mask]

            if count > 0:
                count_index = [[self.max_detection] * candidate_boxes.size(-1)]
                count_updates = [torch.zeros(candidate_boxes.size(-1)).fill_(count)]
                indices = torch.cat((torch.tensor(indices), torch.tensor(count_index)), dim=0)
                updates = torch.cat((torch.stack(updates).cuda(), torch.stack(count_updates).cuda()), dim=0)
                outputs = outputs.cuda().scatter_(0, indices.cuda(), updates)

            return outputs

        valid_count = []
        final_result = []
        combined_boxes = torch.cat((boxes, objectness, landmark_coord), dim=2)

        for combined_box in combined_boxes:
            result = single_batch_nms(combined_box)
            valid_count.append(result[self.max_detection][0].unsqueeze(0).unsqueeze(0))
            final_result.append(result[0:self.max_detection].unsqueeze(0))

        valid_count = torch.cat(valid_count, dim=0).cuda()
        final_result = torch.cat(final_result, dim=0).cuda()

        nms_boxes, nms_scores, nms_landmarks = torch.split(final_result, [4, 1, final_result.size(-1) - 5], dim=-1)
        return nms_boxes, nms_scores, nms_landmarks, valid_count.int()
