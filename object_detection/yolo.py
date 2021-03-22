import torch
from torch import nn as nn
from torch.nn import functional as f

LAMBDA_COORD = 5
NO_OBJ = 0.5


class SimpleYoloModel(nn.Module):
    def __init__(self, inference=False):
        super(SimpleYoloModel, self).__init__()
        self.inference = inference
        self.conv_1 = nn.Conv2d(3, 8, (2, 2), stride=(1, 1))
        self.pool_1 = nn.MaxPool2d(3)
        self.conv_2 = nn.Conv2d(8, 16, (2, 2), stride=(1, 1))
        self.pool_2 = nn.MaxPool2d(3)
        self.conv_3 = nn.Conv2d(16, 32, (2, 2), stride=(1, 1))
        self.pool_3 = nn.MaxPool2d(3)
        self.op_layer = nn.Conv2d(32, 15, (2, 2))

    def forward(self, x):
        x = self.pool_1(f.relu(self.conv_1(x)))
        x = self.pool_2(f.relu(self.conv_2(x)))
        x = self.pool_3(f.relu(self.conv_3(x)))
        x = self.op_layer(x)
        batch, kernels, height, width = x.shape
        x = x.reshape((batch, height, width, int(kernels / 5), 5))
        if self.inference:
            return decode_output(x)


def yolo_localization_loss(output: torch.Tensor, labels: torch.Tensor):
    label_conf, label_xy, label_wh = torch.split(
        labels, (1, 2, 2), dim=-1
    )
    no_obj = (label_conf == 0).to(torch.int)
    op_conf, prediction_xy, prediction_wh = torch.split(
        decode_output(output), (1, 2, 2), dim=-1
    )

    first = LAMBDA_COORD * torch.sum(
        torch.square(prediction_xy - label_xy) * label_conf
    )
    second = LAMBDA_COORD * torch.sum(
        torch.square(prediction_wh - label_wh) * label_conf
    )
    third = torch.sum(
        torch.square(op_conf - label_conf) * label_conf
    )
    fourth = NO_OBJ * torch.sum(
        torch.square(op_conf - label_conf) * no_obj
    )
    return first + second + third + fourth


def iou(box_1, box_2):
    x1, y1, x1_dash, y1_dash = box_1
    x2, y2, x2_dash, y2_dash = box_2
    w_dash = min([x2_dash, x1_dash]) - max([x2, x1])
    h_dash = min([y2_dash, y1_dash]) - max([y2, y1])
    w = max([0, w_dash])
    y = max([0, h_dash])
    intersection = w * y

    total_area = (x2_dash - x2) * (y2_dash - y2) + \
                 (x1_dash - x1) * (y1_dash - y1)
    union = total_area - intersection
    return intersection / union


def decode_output(output: torch.Tensor):
    prediction_confidence = torch.sigmoid(output[..., 0:1])
    prediction_xy = torch.sigmoid(output[..., 1:3])
    prediction_wh = torch.sigmoid(output[..., 3:5])
    return torch.cat([prediction_confidence, prediction_xy,
                      prediction_wh])


def non_max_suppression(output):
    bbox = []
    conf_scores = []

    output_tensor = output.detach().clone()
    for anchor in range(3):

        conf, xy, wh = torch.split(output_tensor[..., anchor, :], (1, 2, 2), dim=-1)
        conf = conf[0, ...].flatten()
        xy, wh = xy[0, ...], wh[0, ...]
        top_left = xy - wh * 0.5
        bottom_right = xy + wh * 0.5
        bbox_tensor = torch.cat([top_left, bottom_right], dim=-1).flatten(
            start_dim=1, end_dim=2
        ).squeeze(0)[:, 0, :]

        bool_flag = torch.zeros(len(conf))

        while conf.sum() > 0:
            max_idx = conf.argmax().item()
            c_max = conf[max_idx]
            conf_scores.append(c_max.item())
            conf[max_idx] = 0
            b_max = bbox_tensor[max_idx]
            bbox.append(b_max)
            for b_idx, b in enumerate(bbox_tensor):
                if iou(b, b_max) > 0.6:
                    bool_flag[b_idx] = 0
                else:
                    bool_flag[b_idx] = 1
            bbox_tensor = bool_flag * bbox_tensor
    return conf_scores, bbox