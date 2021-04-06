import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from object_detection.yolo import CSVDataset, ClassifierBackBone


if torch.cuda.is_available():
    gpu = 1
else:
    gpu = 0

data_set = CSVDataset(
    r'C:\Users\Vishnu\Documents\model_zoo\data\data.csv',
    r'C:\Users\Vishnu\Documents\model_zoo\data\human_data'
)
data_len = len(data_set)
train_len, val_len = int(0.6 * data_len), int(0.2 * data_len)
test_len = data_len - (train_len + val_len)

train_set, val_set, test_set = random_split(
    data_set, (train_len, val_len, test_len)
)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)
test_loader = DataLoader(test_set, batch_size=16)

model = ClassifierBackBone()

trainer = pl.Trainer(precision=32, gpus=gpu)
trainer.fit(model, train_loader, val_loader)
