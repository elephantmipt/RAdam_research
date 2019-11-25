from pathlib import Path
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet34

from source.trainer import Trainer, Config
from source.optimizer import AdamW, RAdam


def init_weights(m):
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.01)


def main():
    with open("source/config.yml", 'r') as file:
        config = Config(**yaml.load(file)['opt_config'])

    cifar_train = CIFAR10('.', train=True, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                         transforms.ToTensor()]),
                          download=True)
    cifar_test = CIFAR10('.', train=False, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                         transforms.ToTensor()]),
                         download=True)

    dl_train = DataLoader(cifar_train, batch_size=16)
    dl_test = DataLoader(cifar_test, batch_size=16)

    logdir = "./logdir/Adam"
    num_epochs = 10

    loaders = {'train': dl_train, 'valid': dl_test}

    model = resnet34()

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam
    runner = Trainer(model=model, config=config, train_loader=dl_train, test_loader=dl_test, loss=criterion,
                     log_dir=Path(logdir))

    for e in range(config.epochs):
        runner.train(e)

    runner.test()

    logdir = "./logdir/AdamW"

    model.apply(init_weights)
    config.optimizer = AdamW
    runner = Trainer(model=model, config=config, train_loader=dl_train, test_loader=dl_test, loss=criterion,
                     log_dir=Path(logdir))

    for e in range(config.epochs):
        runner.train(e)

    runner.test()

    logdir = "./logdir/RAdam"

    model.apply(init_weights)
    config.optimizer = RAdam
    runner = Trainer(model=model, config=config, train_loader=dl_train, test_loader=dl_test, loss=criterion,
                     log_dir=Path(logdir))

    for e in range(config.epochs):
        runner.train(e)

    runner.test()


if __name__ == '__main__':
    main()