import math
import torch
from torch.optim.optimizer import Optimizer, required
from tensorboard import SummaryWriter


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Trainer:

    def __init__(self, model, config, train_loader, test_loader, loss, log_dir):

        self.cuda = config.cuda
        self.device = config.device
        self.seed = config.seed
        self.lr = config.lr
        self.epochs = config.epochs
        self.save_model = config.save_model
        self.batch_size = config.batch_size
        self.log_interval = config.log_interval
        self.loss = loss

        self.globaliter = 0

        torch.manual_seed(self.seed)

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        self.train_loader = train_loader

        self.test_loader = test_loader

        self.model = model
        self.optimizer = config.optimizer(self.model.parameters(), lr=self.lr)
        self.logger = SummaryWriter(log_dir)

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
        for batch_idx, (data, target) in enumerate(self.train_loader):

            self.globaliter += 1
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(data)

            loss = self.loss(predictions, target)
            loss.backward()
            for name, param in self.model.named_parameters():
                if 'bn' not in name:
                    assert(any(param.grad != 0))
                    self.logger.add_histogram('Gradient', param.grad, self.globaliter)

            self.optimizer.step()

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))
                self.tb.save_value('Train Loss', 'train_loss', self.globaliter, loss.item())

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                predictions = self.model(data)

                test_loss += F.nll_loss(predictions, target, reduction='sum').item()
                prediction = predictions.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()

            test_loss /= len(self.test_loader.dataset)
            accuracy = 100. * correct / len(self.test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset), accuracy))