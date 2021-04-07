import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils import tensorboard
import tqdm

from object_detection.yolo import CSVDataset, ClassifierBackBone
from torchmetrics import Accuracy, MetricCollection

import argparse


def main(csv_file, data_path):
    gpu = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data_set = CSVDataset(csv_file, data_path)
    tb_logger = TensorBoardLogger(save_dir='./logs/')

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = torch.nn.BCELoss()

    writer = tensorboard.SummaryWriter('./logs')
    writer.add_graph(model, input_to_model=torch.randn(1, 3, 400, 400))

    train_collection = MetricCollection([Accuracy(compute_on_step=False)])
    val_collection = MetricCollection([Accuracy(compute_on_step=False)])

    for ep in range(1, 1000):
        loss_val = 0
        with tqdm.tqdm(train_loader, unit="batch") as train_epoch:
            for idx, (inp, label) in enumerate(train_epoch):
                train_epoch.set_description(f'Train: {ep}')
                inp, label = inp.to(gpu), label.to(gpu)
                optimizer.zero_grad()
                out = model(inp)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                loss_val += loss.item()
                out, label = torch.round(out).to(int).to('cpu'), label.to(int).to('cpu')
                train_collection(out, label)
                train_epoch.set_postfix(loss=loss_val / idx)
            writer.add_scalars('Training', train_collection.compute(), ep)
            train_collection.reset()

        val_loss_val = 0
        with tqdm.tqdm(val_loader, unit="batch") as val_epoch:
            for idx, (inp, label) in enumerate(val_epoch):
                with torch.no_grad():
                    inp, label = inp.to(gpu), label.to(gpu)
                    out = model(inp)
                    loss = criterion(out, label)
                    val_loss_val += loss.item()
                    out, label = torch.round(out).to(int).to('cpu'), label.to(int).to('cpu')
                    val_collection(out, label)
                    train_epoch.set_postfix(loss=loss_val / idx)
        writer.add_scalars('Validation', val_collection.compute(), ep)
        val_collection.reset()
        writer.add_scalars('Loss', {
            'training': loss_val / len(train_loader),
            'validation': val_loss_val / len(val_loader)
        }, ep)
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('data_path')
    args = parser.parse_args()
    main(args.csv_file, args.data_path)
