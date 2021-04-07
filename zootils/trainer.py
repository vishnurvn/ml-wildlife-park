import torch
import tqdm
from torchmetrics import MetricCollection

from collections import OrderedDict


class Trainer:
    def __init__(
            self,
            gpu='cpu',
            epochs=100,
            metrics_collection=[]
    ):
        self.gpu = gpu
        self.epochs = epochs
        self.metric_collection = metrics_collection

    def fit(self, model, train_loader, val_loader):
        model.to(self.gpu)
        criterion = model.config_loss()
        optimizer = model.config_optimizer()

        for ep in range(1, self.epochs + 1):
            with tqdm.tqdm(train_loader, unit="batch") as train_epoch:
                for inp, label in train_epoch:
                    train_epoch.set_description(f"Epoch: {ep}")
                    inp, label = inp.to(self.gpu), label.to(self.gpu)
                    optimizer.zero_grad()
                    out = model(inp)
                    loss = criterion(out, label)
                    loss.backward()
                    optimizer.step()
                    if self.metric_collection is not None:
                        train_epoch.set_postfix(self.train_metrics)

            with torch.no_grad():
                with tqdm.tqdm(val_loader, unit="batch") as val_epoch:
                    for inp, label in val_epoch:
                        val_epoch.set_description(f"Epoch: {ep}")
                        inp, label = inp.to(self.gpu), label.to(self.gpu)
                        out = model(inp)
                        loss = criterion(out, label)
                        if self.metric_collection is not None:
                            val_epoch.set_postfix(self.val_metrics)

    def test(self, model, test_loader):
        model.to(self.gpu)
        criterion = model.config_loss()

        with torch.no_grad():
            for inp, label in test_loader:
                inp, label = inp.to(self.gpu), label.to(self.gpu)
                out = model(inp)
                loss = criterion(out, label)
