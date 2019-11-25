from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as d
from tensorboardX import SummaryWriter


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Trainer:

    def __init__(self, model: nn.Module, config: Config,
                 train_loader: d.DataLoader, test_loader: d.DataLoader,
                 loss: nn.Module, log_dir: Path):

        self.cuda = config.cuda
        self.device = config.device
        self.seed = int(config.seed)
        self.lr = float(config.lr)
        self.epochs = int(config.epochs)
        self.batch_size = int(config.batch_size)
        self.log_interval = config.log_interval
        self.loss = loss

        self.globaliter = 0

        torch.manual_seed(self.seed)

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        self.train_loader = train_loader

        self.test_loader = test_loader

        self.model = model
        self.optimizer = config.optimizer(self.model.parameters(), lr=self.lr)
        self.logger = SummaryWriter(log_dir.as_posix())

        self.model.to(self.device)

    def change_conf(self, config):
        self.cuda = config.cuda
        self.device = config.device
        self.seed = config.seed
        self.lr = config.lr
        self.epochs = config.epochs
        self.save_model = config.save_model
        self.batch_size = config.batch_size
        self.log_interval = config.log_interval
        self.optimizer = config.optimizer(self.model.parameters(), lr=self.lr)

    def train(self, epoch):

        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}: loss={0}",
                    total=len(self.train_loader.dataset) // self.batch_size)
        for data, target in pbar:

            self.globaliter += 1
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(data)

            loss = self.loss(predictions, target)
            loss.backward()
            for name, param in self.model.named_parameters():
                if 'bn' not in name:
                    self.logger.add_histogram('Gradient',
                                              param.grad, self.globaliter)

            self.optimizer.step()

            pbar.set_description(desc=f"Train epoch {epoch}: loss={loss.item():.6f}")
            self.logger.add_scalar('Train Loss', loss.item(), self.globaliter)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                predictions = self.model(data)

                test_loss += self.loss(predictions, target).item()
                pred = predictions.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.test_loader.dataset)
            accuracy = 100. * correct / len(self.test_loader.dataset)

            print(f'Test set: Average loss: {test_loss:.4f},'
                  f' Accuracy: {correct}/{len(self.test_loader.dataset)}'
                  f' ({accuracy:.0f}%)')
