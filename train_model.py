import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import tqdm

from object_detection.yolo import SimpleYoloModel, yolo_localization_loss, iou, non_max_suppression
from data_loader import YoloDataset

LOG_DIR = './object_detection/logs/yolo'


model = SimpleYoloModel()
data_set = YoloDataset('./data/1_annots')
data_len = len(data_set)
train_len = int(data_len * 0.6)
val_len = int(data_len * 0.2)
test_len = data_len - (train_len + val_len)
train_set, val_set, test_set = random_split(data_set, lengths=(train_len, val_len, test_len))

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=0.001)


def calc_batch_iou(label_bbox, output_bbox):
    iou_val, l_idx, op_idx = 0, 1, 1
    for l_idx, l_bbox in enumerate(label_bbox, start=1):
        for op_idx, op_bbox in enumerate(output_bbox, start=1):
            iou_val += iou(l_bbox, op_bbox)
            print(iou_val)
    return iou_val / (l_idx * op_idx)


def train(epochs):
    print('Start training')
    writer = SummaryWriter(LOG_DIR)
    for ep in range(epochs):
        loss_val = 0
        epoch_iou = 0
        idx = 0
        with tqdm.tqdm(train_loader, unit="batch") as t_epoch:
            for idx, (label, inp) in enumerate(t_epoch, start=1):
                t_epoch.set_description(f"Epoch: {ep}")
                optimizer.zero_grad()
                op = model(inp)
                loss = yolo_localization_loss(op, label)
                _, op_bbox = non_max_suppression(op, 0.6)
                epoch_iou += calc_batch_iou(label[..., :][label[..., 0] == 1][:, 1:], op_bbox)
                loss.backward()
                optimizer.step()
                loss_val += loss.item()
                t_epoch.set_postfix(loss=loss_val / idx)

        val_loss_value = 0
        val_epoch_iou = 0
        val_idx = 0
        with torch.no_grad():
            for val_idx, (label, inp) in enumerate(val_loader):
                op = model(inp)
                loss = yolo_localization_loss(op, label)
                _, op_bbox = non_max_suppression(op, 0.6)
                val_epoch_iou += calc_batch_iou(label[..., :][label[..., 0] == 1][:, 1:], op_bbox)
                val_loss_value += loss.item()

        writer.add_scalars('Loss', {
            'train': loss_val / idx,
            'validation': val_loss_value / val_idx
        }, ep)
        writer.add_scalars('IOU', {
            'train': epoch_iou / idx,
            'validation': val_epoch_iou / val_idx
        }, ep)
    writer.close()


if __name__ == '__main__':
    train(100)
