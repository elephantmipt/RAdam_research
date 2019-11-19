import torch
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet34

from catalyst import dl


from .optim import AdamW, RAdam


def main():
    cifar_train = CIFAR10('.', train=True, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                         transforms.ToTensor()]),
                          download=True)
    cifar_test = CIFAR10('.', train=False, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                         transforms.ToTensor()]),
                         download=True)

    dl_train = DataLoader(cifar_train, batch_size=16)
    dl_test = DataLoader(cifar_test, batch_size=16)

    logdir = "./logdir"
    num_epochs = 10

    loaders = {'train': dl_train, 'valid': dl_test}

    model = resnet34().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = dl.SupervisedRunner()

    runner.train(model=model, criterion=criterion, optimizer=optimizer, loaders=loaders, num_epochs=num_epochs,
                 verbose=True, logdir=logdir)



