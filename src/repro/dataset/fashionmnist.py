from collections import OrderedDict

import torch
from torchvision import datasets, transforms


DATA_SIZE = 50000


def build(batch_size, data_path, num_workers, mini=False):

    transformations = [transforms.ToTensor()]

    if mini:
        transformations.insert(0, transforms.Resize(7))

    dataset = datasets.FashionMNIST(
        data_path, train=True, download=True, transform=transforms.Compose(transformations))
    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(min(DATA_SIZE, len(dataset))))

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(DATA_SIZE, len(dataset)))

    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    dataset = datasets.FashionMNIST(
        data_path, train=False, download=True, transform=transforms.Compose(transformations))
    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(min(DATA_SIZE, len(dataset))))

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return OrderedDict(train=train_loader, valid=valid_loader, test=test_loader)


def add_subparser(parser):
    fashionmnist_parser = parser.add_parser(
        'fashionmnist', help='Arguments for dataset fashionmnist')

    return fashionmnist_parser
