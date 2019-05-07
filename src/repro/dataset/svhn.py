from collections import OrderedDict

import torch
from torchvision import datasets, transforms


def build(data_path, mini=False):

    transformations = [transforms.ToTensor()]

    if mini:
        transformations.insert(0, transforms.Resize(8))

    transform = transforms.Compose(transformations)

    train_dataset = datasets.SVHN(
        data_path, split='train', download=True,
        transform=transform)

    test_dataset = datasets.SVHN(
        data_path, split='test', download=True,
        transform=transform)

    return OrderedDict(dataset=torch.utils.data.ConcatDataset([train_dataset, test_dataset]))
