from collections import OrderedDict

import torch
from torchvision import datasets, transforms


DATA_SIZE = 45000


def build(batch_size, data_path, num_workers, mini=False):
    transformations = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    if mini:
        transformations.insert(0, transforms.Resize(8))

    trainset = datasets.CIFAR100(root=data_path, train=True,
                                 download=True, transform=transforms.Compose(transformations))

    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(min(DATA_SIZE, len(trainset))))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=sampler, num_workers=num_workers)

    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(DATA_SIZE, len(trainset)))

    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=sampler, num_workers=num_workers)

    testset = datasets.CIFAR100(root=data_path, train=False,
                                download=True, transform=transforms.Compose(transformations))

    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(min(DATA_SIZE, len(testset))))

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              sampler=sampler, num_workers=num_workers)

    return OrderedDict(train=train_loader, valid=valid_loader, test=test_loader)
