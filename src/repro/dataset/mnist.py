from collections import OrderedDict

import torch
from torchvision import datasets, transforms


DATA_SIZE = 50000


def build(batch_size, data_path, fold=1, mini=False):

    transformations = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]

    if mini:
        transformations.insert(0, transforms.Resize(7))

    dataset = datasets.MNIST(
        data_path, train=True, download=True, transform=transforms.Compose(transformations))
    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(min(DATA_SIZE, len(dataset))))

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(DATA_SIZE, len(dataset)))

    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    dataset = datasets.MNIST(
        data_path, train=False, download=True, transform=transforms.Compose(transformations))
    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(min(DATA_SIZE, len(dataset))))

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    return OrderedDict(train=train_loader, valid=valid_loader, test=test_loader)
