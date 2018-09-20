import os
import torch
import torchvision
import torchvision.transforms as transforms


def get_ZCA(data, epsilon):
    mu = data.mean(dim=0)
    data = data - mu
    cov = torch.mm(data.t(), data) / data.shape[0]
    e, v = torch.symeig(cov, eigenvectors=True)
    e = torch.diag(1. / (e + epsilon) ** 0.5)
    icov = torch.mm(torch.mm(v, e), v.t())
    return mu, icov


def apply_ZCA(data, mu, icov):
    return torch.mm(data - mu.view(1, -1) , icov)


def whiten_mnist(data_path, out_path, epsilon):
    out_file = os.path.join(out_path, 'mnist_eps=' + str(epsilon) + '.t7')
    if os.path.exists(out_file):
        print(out_file + ' already exists!')
        return out_file
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Prepare the loaders
    trainset = torchvision.datasets.MNIST(root=data_path, train=True,
                                          download=False,
                                          transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root=data_path, train=False,
                                         download=False,
                                         transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000)

    # Create the train/valid/test splits
    for d, l in trainloader:
        data, labels = d, l
    s = data.shape
    train_data = data[:50000].view(50000, -1)
    train_labels = labels[:50000]
    valid_data = data[50000:60000].view(10000, -1)
    valid_labels = labels[50000:60000]
    for d, l in testloader:
        test_data, test_labels = d, l
    test_data = test_data.view(10000, -1)

    # Get the ZCA parameters
    mu, icov = get_ZCA(train_data, epsilon)

    # Apply ZCA
    train_data = apply_ZCA(train_data, mu, icov)
    valid_data = apply_ZCA(valid_data, mu, icov)
    test_data = apply_ZCA(test_data, mu, icov)

    # Reshape
    train_data = train_data.view(-1, *s[1:])
    valid_data = valid_data.view(-1, *s[1:])
    test_data = test_data.view(-1, *s[1:])

    # Package
    data = {'train': (train_data, train_labels),
            'valid': (valid_data, valid_labels),
            'test': (test_data, test_labels)}

    # Save
    torch.save(data, out_file)
    return out_file


class WhiteMNIST(torch.utils.data.TensorDataset):
    """MNIST Dataset for the data created by the code above."""

    def __init__(self, data_path, split):
        self.tensors = torch.load(data_path)[split]


def build(batch_size, data_path, num_workers, epsilon):
    out_file = whiten_mnist(data_path, data_path, epsilon)

    trainset = WhiteMNIST(out_file, 'train')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    validset = WhiteMNIST(out_file, 'valid')
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    testset = WhiteMNIST(out_file, 'test')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    return OrderedDict(train=train_loader, valid=valid_loader, test=test_loader)


if __name__ == '__main__':

    data_path = '/data/milatmp1/laurent/data/'
    out_path = '/Tmp/laurent/test/'
    epsilon = 1e-1

    out_file = whiten_mnist(data_path, out_path, epsilon)

    dataset = WhiteMNIST(out_file, 'train')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1000)

    for i, (features, targets) in enumerate(trainloader):
        print(features.shape, targets.shape)
        print(features.min().item(), features.mean().item(),
              features.max().item())
        features = features.view(features.shape[0], -1)
        c = torch.mm(features.t(), features) / features.shape[0]
        print(((c - torch.eye(c.shape[0]))**2).sum())
        break
