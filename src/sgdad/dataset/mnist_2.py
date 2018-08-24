import torch
from torchvision import datasets, transforms

def build(train_batch_size=128, test_batch_size=500, num_workers=2):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader, test_loader


def add_subparser(parser):
    mnist_parser = parser.add_parser('mnist', help='Arguments for dataset mnist')

    return mnist_parser

