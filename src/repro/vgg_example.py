#!/usr/bin/env python
import os
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

try:
    from apex import amp
    APEX_AVALIABLE = True
except ImportError:
    APEX_AVALIABLE = False


class VGG11(nn.Module):

    def __init__(self, dropout, bn=False):
        super(VGG11, self).__init__()
        dims = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        modules = []
        in_dim = 3
        for i, dim in enumerate(dims):
            if dim == 'M':
                modules += [nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Dropout2d(dropout, inplace=True)]
            else:
                if bn:
                    modules += [nn.Conv2d(in_dim, dim, kernel_size=3,
                                          padding=1, bias=False),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Conv2d(in_dim, dim, kernel_size=3,
                                          padding=1, bias=True),
                                nn.ReLU(inplace=True)]
                in_dim = dim
        modules += [nn.Conv2d(in_dim, 10, kernel_size=1, padding=0,
                              bias=True)]
        self.net = nn.Sequential(*modules)
        self.kaiming_init()

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)

    def kaiming_init(self):
        for m in self.net:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    torch.nn.init.constant_(m.weight, 1.0)
                    torch.nn.init.constant_(m.bias, 0.0)


def train_loop(net, optimizer, loader, device):
    """Loop for network training."""
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        if APEX_AVALIABLE:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # Monitoring
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        train_loss_norm = train_loss / (batch_idx + 1)
        ratio = float(total - correct) / total
    return train_loss_norm, ratio


@torch.no_grad()
def eval_loop(net, loader, device):
    """Loop for network evalutaion."""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        # Monitoring
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        test_loss_norm = test_loss / (batch_idx + 1)
        ratio = float(total - correct) / total
    return test_loss_norm, ratio


def main(device=None, seed=1111, lr=0.01, decay=0.1, bs=100, num_workers=0,
         data_path=os.path.join(os.environ['SLURM_TMPDIR'], 'data'), id=None, 
         xp_path=os.environ['SLURM_TMPDIR'], nepochs=300, callback=None,
         dropout=0.0, bn=False):

    os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[1]
    device = 'cuda'

    # -------------------------------------------------------------------------
    # Preparing Model and Optimizer

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = VGG11(dropout, bn).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4)

    if APEX_AVALIABLE:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O3',
                                        keep_batchnorm_fp32=True,
                                        verbosity=0)

    if os.path.exists(xp_path):
        checkpoint = torch.load(xp_path, map_location=device)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_err = checkpoint['best_err']
        start_epoch = checkpoint['epoch']
    else:
        best_err = 1
        start_epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60,
                                                gamma=decay,
                                                last_epoch=start_epoch-1)

    # -------------------------------------------------------------------------
    # Preparing Data Streams

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_set = CIFAR10(root=data_path, train=True, download=False,
                        transform=transform_train)
    valid_set = CIFAR10(root=data_path, train=True, download=False,
                        transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=bs,
                              sampler=SubsetRandomSampler(range(45000)),
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=bs,
                              sampler=SubsetRandomSampler(range(45000, 50000)),
                              num_workers=num_workers, pin_memory=True)

    # -------------------------------------------------------------------------
    # Main Loop

    for epoch in range(start_epoch + 1, nepochs + 1):
        torch.manual_seed(seed + epoch)
        torch.cuda.manual_seed_all(seed + epoch)
        random.seed(seed + epoch)

        # Learning Rate Decay
        scheduler.step()

        # Actual training and validation loops
        train_l, train_m = train_loop(net, optimizer, train_loader,
                                      device)
        valid_l, valid_m = eval_loop(net, valid_loader, device)
        if valid_m < best_err:
            best_err = valid_m

        torch.save({'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_err': best_err,
                    'epoch': epoch}, xp_path + '_tmp')
        os.rename(xp_path + '_tmp', xp_path)
        
        if callback is not None:
            callback(step=epoch, objective=valid_m, finished=False)

    if callback is not None:
        callback(step=epoch, objective=valid_m, finished=True)
