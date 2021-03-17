import torch
import torch.nn as nn
import torch.nn.functional as f


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
        return x.reshape((batch, int(kernels / 5), 5, height, width))


def yolo_localization_loss(output: torch.Tensor, labels: torch.Tensor):
    op_conf = torch.sigmoid(output[:, :, 0, :, :])
    label_conf = labels[:, :, 0, :, :]
    no_obj = (label_conf == 0).to(torch.int)
    x_i_hat, x_i = torch.sigmoid(output[:, :, 1, :, :]), torch.sigmoid(labels[:, :, 1, :, :])
    y_i_hat, y_i = torch.sigmoid(output[:, :, 2, :, :]), torch.sigmoid(labels[:, :, 2, :, :])

    w_i_hat, w_i = torch.sigmoid(output[:, :, 3, :, :]), torch.sigmoid(labels[:, :, 3, :, :])
    h_i_hat, h_i = torch.sigmoid(output[:, :, 4, :, :]), torch.sigmoid(labels[:, :, 4, :, :])

    first = LAMBDA_COORD * torch.sum(
        (torch.square(x_i - x_i_hat) + torch.square(y_i - y_i_hat)) * label_conf
    )
    second = LAMBDA_COORD * torch.sum(
        (torch.square(
            torch.sqrt(w_i) - torch.sqrt(w_i_hat)
        ) + torch.square(
            torch.sqrt(h_i) - torch.sqrt(h_i_hat))
        ) * label_conf
    )
    third = torch.sum(
        torch.square(op_conf - label_conf) * label_conf
    )
    fourth = NO_OBJ * torch.sum(
        torch.square(op_conf - label_conf) * no_obj
    )
    return first + second + third + fourth


def iou(box_1, box_2):
    pass


def decode_output(output):
    pass
