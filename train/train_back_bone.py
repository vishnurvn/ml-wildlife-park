import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger

from object_detection.yolo import CSVDataset, ClassifierBackBone

import argparse


def main(csv_file, data_path):
    if torch.cuda.is_available():
        gpu = 1
    else:
        gpu = 0

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

    tb_logger.log_graph(model, input_array=torch.randn(1, 3, 400, 400))
    trainer = pl.Trainer(
        precision=32,
        gpus=gpu,
        logger=tb_logger
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('data_path')
    args = parser.parse_args()
    main(args.csv_file, args.data_path)
