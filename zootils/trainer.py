class Trainer:
    def __init__(self, gpu):
        self.gpu = gpu

    def fit(self, model, train_loader, val_loader):
        criterion = model.config_loss()
        optimizer = model.config_optimizer()

        for inp, label in train_loader:
            inp, label = inp.to_device(self.gpu), label.to_device(self.gpu)
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, label)

    def test(self, model, test_loader):
        pass
