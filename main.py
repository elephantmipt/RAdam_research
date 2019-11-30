from pathlib import Path
import yaml

import torch

import shutil

from torch.utils.data import DataLoader


from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18

from source.trainer import Trainer, Config
from source.optimizer import AdamW, RAdam


def init_weights(m):
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.01)


def main():
    with open("source/config.yml", 'r') as file:
        config = Config(**yaml.load(file)['opt_config'])
        scheduler_config = Config
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    cifar_train = CIFAR10('.', train=True, transform=transform_train,
                          download=True)
    cifar_test = CIFAR10('.', train=False, transform=transform_test,
                         download=True)

    dl_train = DataLoader(cifar_train, batch_size=config.batch_size)
    dl_test = DataLoader(cifar_test, batch_size=config.batch_size)
    try:
        shutil.rmtree('./logdir')
        print('Previous logs deleted')
    except:
        print('There is no previous logs')
        
    criterion = torch.nn.CrossEntropyLoss()

    print('Training with Adam optimizer...')
    torch.cuda.empty_cache()
    model = resnet18().cuda()

    model.train()
    logdir = "./logdir/Adam"

    config.optimizer = torch.optim.Adam
    runner = Trainer(model=model, config=config, train_loader=dl_train,
                     test_loader=dl_test, loss=criterion, log_dir=Path(logdir))

    for e in range(config.epochs):
        runner.train(e)
        runner.test(e)
        if runner.globaliter >= 100:
            break

    print('Training with AdamW optimizer...')
    torch.cuda.empty_cache()
    model = resnet18().cuda()

    model.train()
    logdir = "./logdir/AdamW"

    config.optimizer = AdamW
    runner = Trainer(model=model, config=config, train_loader=dl_train,
                     test_loader=dl_test, loss=criterion, log_dir=Path(logdir))

    for e in range(config.epochs):
        runner.train(e)
        runner.test(e)
        if runner.globaliter >= 100:
            break

    print('Training with RAdam optimizer...')
    del model
    torch.cuda.empty_cache()
    model = resnet18().cuda()

    model.train()
    logdir = "./logdir/RAdam"

    config.optimizer = RAdam
    runner = Trainer(model=model, config=config, train_loader=dl_train,
                     test_loader=dl_test, loss=criterion, log_dir=Path(logdir))

    for e in range(config.epochs):
        runner.train(e)
        runner.test(e)
        if runner.globaliter >= 100:
            break

    print('Training with SGD optimizer...')
    del model
    torch.cuda.empty_cache()
    model = resnet18().cuda()

    model.train()
    logdir = "./logdir/SGD"
    config.optimizer = torch.optim.SGD(momentum=0.9)
    runner = Trainer(model=model, config=config, train_loader=dl_train,
                     test_loader=dl_test, loss=criterion, log_dir=Path(logdir))
    for e in range(config.epochs):
        runner.train(e)
        runner.test(e)
        if runner.globaliter >= 100:
            break


if __name__ == '__main__':
    main()
