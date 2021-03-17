import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import SimpleYoloModel, yolo_localization_loss
from data_loader import YoloDataset


model = SimpleYoloModel()
data_set = YoloDataset('./data/1_annots')
data_len = len(data_set)
train_len = int(data_len * 0.7)
test_len = data_len - train_len
train_set, test_set = random_split(data_set, lengths=(train_len, test_len))


train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=0.001)


def train(epochs):
    for ep in range(epochs):
        loss_val = 0
        for label, inp in train_loader:
            optimizer.zero_grad()
            op = model(inp)
            loss = yolo_localization_loss(op, label)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        print(f"Epoch: {ep}; Loss: {loss_val}")


if __name__ == '__main__':
    train(3)
