import torch
from torchvision import datasets, transforms

def build(train_batch_size=128, test_batch_size=500, num_workers=2):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data/cifar10', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=num_workers)
    
    testset = datasets.CIFAR10(root='./data/cifar10', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def add_subparser(parser):
    cifar10_parser = parser.add_parser('cifar10', help='Arguments for dataset cifar10')

    return cifar10_parser

