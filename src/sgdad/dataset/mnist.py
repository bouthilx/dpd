import torch
from torchvision import datasets, transforms


def build(batch_size=128, data_size=50000):

    dataset = datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(min(data_size, len(dataset))))

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler)

    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(data_size, len(dataset)))

    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler)

    dataset = datasets.MNIST(
            '../data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
    sampler = torch.utils.data.sampler.SubsetRandomSampler(range(min(data_size, len(dataset))))

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler)

    return train_loader, valid_loader, test_loader
