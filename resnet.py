import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet34

from catalyst import dl
from catalyst.dl.callbacks import AccuracyCallback


from optim import AdamW, RAdam

from source.logger import logger


def init_weights(m):
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.01)


def main():
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
    for name, param in model.named_parameters():
        param.requires_grad = True

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = dl.SupervisedRunner()

    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, num_epochs=num_epochs,
                 verbose=True, logdir=logdir, callbacks=[logger.TensorboardLogger(),
                                                         AccuracyCallback(num_classes=10)],)

    logdir = "./logdir/AdamW"

    model.apply(init_weights)
    optimizer = AdamW()
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, num_epochs=num_epochs,
                 verbose=True, logdir=logdir, callbacks=[logger.TensorboardLogger(),
                                                         AccuracyCallback(num_classes=10)], )

    logdir = "./logdir/RAdam"

    model.apply(init_weights)
    optimizer = RAdam()
    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, num_epochs=num_epochs,
                 verbose=True, logdir=logdir, callbacks=[logger.TensorboardLogger(),
                                                         AccuracyCallback(num_classes=10)], )


if __name__ == '__main__':
    main()