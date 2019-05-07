from collections import OrderedDict

import torch
from torchvision import datasets, transforms


def build(data_path, mini=False):
    transformations = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    if mini:
        transformations.insert(0, transforms.Resize(8))

    transform = transforms.Compose(transformations)

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    return OrderedDict(dataset=torch.utils.data.ConcatDataset([train_dataset, test_dataset]))
