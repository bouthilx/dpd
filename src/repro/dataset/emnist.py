from collections import OrderedDict

import torch
from torchvision import datasets, transforms


def build(data_path):

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.EMNIST(data_path, train=True, download=True, split='balanced',
                                    transform=transform)

    valid_dataset = datasets.EMNIST(data_path, train=False, download=True, split='balanced',
                                    transform=transform)

    return OrderedDict(dataset=torch.utils.data.ConcatDataset([train_dataset, test_dataset]))
