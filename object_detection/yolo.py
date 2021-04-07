import csv
import os
from collections import Counter

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import nn as nn
from torch.nn import functional as f
from torch.utils.data import Dataset
from torchvision.transforms import transforms as t
from torchmetrics import Accuracy, Precision, Recall, MetricCollection
from torch.utils.tensorboard import SummaryWriter


LAMBDA_COORD = 5
NO_OBJ = 0.5

transforms = t.Compose([
    t.ToTensor(),
    t.Normalize((1, 1, 1), (1, 1, 1))
])


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, (1, 1))
        self.activation_layer = nn.ReLU()

    def forward(self, x):
        op = self.block(x)
        if self.in_channel != self.out_channel:
            return self.activation_layer(self.conv1x1(x) + op)
        return self.activation_layer(op + x)


class ResidualBottleneck(nn.Module):
    def __init__(self, in_channel, stride):
        super(ResidualBottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (3, 3), padding=(1, 1), stride=stride),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, (1, 1), stride=stride)
        self.activation_layer = nn.ReLU()

    def forward(self, x):
        op = self.block(x)
        return self.activation_layer(self.conv1x1(x) + op)


class SimpleYoloModel(nn.Module):
    def __init__(self, inference=False):
        super(SimpleYoloModel, self).__init__()
        self.inference = inference
        self.conv_1 = nn.Conv2d(3, 8, (2, 2), stride=(1, 1))
        self.pool_1 = nn.MaxPool2d(3)
        self.conv_2 = nn.Conv2d(8, 64, (2, 2), stride=(1, 1))
        self.pool_2 = nn.MaxPool2d(3)
        self.conv_3 = nn.Conv2d(64, 128, (2, 2), stride=(1, 1))
        self.pool_3 = nn.MaxPool2d(3)
        self.op_layer = nn.Conv2d(128, 15, (2, 2))

    def forward(self, x):
        x = self.pool_1(f.relu(self.conv_1(x)))
        x = self.pool_2(f.relu(self.conv_2(x)))
        x = self.pool_3(f.relu(self.conv_3(x)))
        x = self.op_layer(x)
        batch, kernels, height, width = x.shape
        x = x.reshape((batch, height, width, int(kernels / 5), 5))
        if self.inference:
            return decode_output(x)
        return x


class ClassifierBackBone(pl.LightningModule):
    def __init__(self):
        super(ClassifierBackBone, self).__init__()
        self.back_bone = nn.Sequential(
            nn.Conv2d(3, 32, (7, 7), stride=(2, 2)),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64),
            ResidualBottleneck(64, 2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            ResidualBottleneck(128, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),
            ResidualBottleneck(256, 2),
            ResidualBlock(256, 256),
            nn.Flatten(),
            nn.Linear(256 * 25 * 25, 1),
            nn.Sigmoid()
        )
        self.criterion = torch.nn.BCELoss()
        self.train_metrics = MetricCollection({
            'train_accuracy': Accuracy(compute_on_step=False),
            'train_precision': Precision(compute_on_step=False),
            'train_recall': Recall(compute_on_step=False),
        })
        self.val_metrics = MetricCollection({
            'val_accuracy': Accuracy(compute_on_step=False),
            'val_precision': Precision(compute_on_step=False),
            'val_recall': Recall(compute_on_step=False)
        })

    def forward(self, x):
        return self.back_bone(x)

    def configure_optimizers(self):
        optimizer_func = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer_func

    def training_step(self, train_batch, batch_idx):
        inp, label = train_batch
        out = self.back_bone(inp)
        loss = self.criterion(out, label)
        out = torch.round(out).to(int).to('cpu')
        label = label.to(int).to('cpu')
        self.train_metrics(out, label)
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, val_batch, batch_idx):
        inp, label = val_batch
        out = self.back_bone(inp)
        loss = self.criterion(out, label)
        out = torch.round(out).to(int).to('cpu')
        label = label.to(int).to('cpu')
        self.val_metrics(out, label)
        self.log("val_loss", loss.item(), prog_bar=True)
        return loss

    def on_train_epoch_end(self, outputs):
        metrics = self.train_metrics.compute()
        self.logger.experiment.add_scalars('Train', metrics)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.logger.experiment.add_scalars('Validation', metrics)


class CSVDataset(Dataset):
    """
    File format. 2 Columns (folder/pic_name, label)
    Arguments:
        file_path: folder/pic_name
        data_path: path to the folder

    """
    def __init__(self, file_path, data_path):
        super(CSVDataset, self).__init__()
        self.data_path = data_path
        self.data = []
        with open(file_path, 'r') as fp:
            reader = csv.DictReader(fp, fieldnames=('file_path', 'label'))
            for line in reader:
                self.data.append(line)

    def __getitem__(self, item):
        image_path = self.data[item]['file_path']
        image = Image.open(os.path.join(self.data_path, image_path))
        label = float(self.data[item]['label'])

        input_tensor = transforms(image)
        return input_tensor, torch.tensor([label])

    def __len__(self):
        return len(self.data)

    def label_counts(self):
        label_list = []
        for item in self.data:
            label_list.append(item['label'])
        counter = Counter(label_list)
        return counter


def yolo_localization_loss(output: torch.Tensor, labels: torch.Tensor):
    label_conf, label_xy, label_wh = torch.split(labels, (1, 2, 2), dim=-1)
    no_obj = (label_conf == 0).to(torch.int)
    op_conf, prediction_xy, prediction_wh = torch.split(decode_output(output), (1, 2, 2), dim=-1)

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
                      prediction_wh], dim=-1)


def non_max_suppression(output, conf_threshold):
    bbox = []
    conf_scores = []

    output_tensor = output.detach().clone()
    for anchor in range(3):
        conf, xy, wh = torch.split(output_tensor[..., anchor, :], (1, 2, 2), dim=-1)
        conf = conf[0, ...].flatten()
        conf[conf < conf_threshold] = 0
        xy, wh = xy[0, ...], wh[0, ...]
        top_left = xy - wh * 0.5
        bottom_right = xy + wh * 0.5
        bbox_tensor = torch.cat([top_left, bottom_right], dim=-1).flatten(start_dim=0, end_dim=1)

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
            conf = bool_flag * conf
    return conf_scores, bbox
