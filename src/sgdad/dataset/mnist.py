import torch
from torchvision import datasets, transforms


def build(batch_size=128, data_size=50000):
    dataset = datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(min(data_size, len(dataset))))

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size, shuffle=sampler is None, sampler=sampler, **kwargs)

    return train, valid, test


def add_subparser(parser):
    mnist_parser = parser.add_parser('mnist', help='Arguments for dataset mnist')

    return mnist_parser
