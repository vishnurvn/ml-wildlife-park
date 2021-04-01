import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

from object_detection.yolo import YoloBodyDataset, ClassifierBackBone


data_set = YoloBodyDataset(
    r'C:\Users\Vishnu\Documents\model_zoo\data\data.csv',
    r'C:\Users\Vishnu\Documents\model_zoo\data'
)
data_len = len(data_set)
train_len, val_len = int(0.6 * data_len), int(0.2 * data_len)
test_len = data_len - (train_len + val_len)

train_set, val_set, test_set = random_split(
    data_set, (train_len, val_len, test_len)
)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)
test_loader = DataLoader(test_set, batch_size=8)

model = ClassifierBackBone()

trainer = pl.Trainer(precision=32)
trainer.fit(model, train_loader, val_loader)