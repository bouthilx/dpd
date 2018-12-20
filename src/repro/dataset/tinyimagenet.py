from collections import OrderedDict
import os
import urllib
import zipfile
import time

import tqdm

import torch
from torchvision import datasets, transforms

# download-url: http://cs231n.stanford.edu/tiny-imagenet-200.zip


# AlexNet and VGG should be treated differently
#         # DataParallel will divide and allocate batch_size to all available GPUs
#         if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
#             model.features = torch.nn.DataParallel(model.features)
#             model.cuda()
#         else:
#             model = torch.nn.DataParallel(model).cuda()

# Because last fc layer is big and not suitable for DataParallel
# Source: https://github.com/pytorch/examples/issues/144

# Train: 100000
# Val:    10000
# Train:  10000

DIRNAME = 'tiny-imagenet-200'


def download(data_path):
    tinyimagenet_path = os.path.join(data_path, DIRNAME)

    if os.path.isdir(tinyimagenet_path):
        return

    filename = 'tiny-imagenet-200.zip'
    zipfile_path = os.path.join(data_path, filename)

    if os.path.exists(zipfile_path):
        print("Zip File {} found but directory not found. Waiting for concurrent process "
              "to complete downloading and unzipping".format(zipfile_path))
        total_time = 0
        while not os.path.isdir(tinyimagenet_path) and total_time < 600:
            time.sleep(30)
            total_time += 30

        if not os.path.isdir(tinyimagenet_path):
            raise RuntimeError("Zip File {} found but directory not found. Try manually unzipping "
                               "the zip file or delete it if corrupted.".format(zipfile_path))

        return

    # download 
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    u = urllib.request.urlopen(url)
    with open(zipfile_path, 'wb') as f:
        file_size = int(dict(u.getheaders())['Content-Length']) / (10.0**6)
        print("Downloading: {} ({}MB)".format(zipfile_path, file_size))

        file_size_dl = 0
        block_sz = 8192
        pbar = tqdm.tqdm(total=file_size, desc='TinyImageNet')
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            f.write(buffer)
            pbar.update(len(buffer) / (10.0 ** 6))
            # file_size_dl += len(buffer)
            # status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            # status = status + chr(8)*(len(status)+1)
            # print(status)

        pbar.close()

    print("Unzipping files...")
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print("Done")


def build(batch_size, data_path, num_workers):
    # Data loading code
    train_dirpath = os.path.join(data_path, DIRNAME, 'train')
    val_dirpath = os.path.join(data_path, DIRNAME, 'val')
    test_dirpath = os.path.join(data_path, DIRNAME, 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    download(data_path)

    dataset = datasets.ImageFolder(
        train_dirpath,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=False, shuffle=True)

    dataset = datasets.ImageFolder(
        val_dirpath,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
        shuffle=True)

    dataset = datasets.ImageFolder(
            test_dirpath, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
        shuffle=True)

    return OrderedDict(train=train_loader, valid=valid_loader, test=test_loader)

# 
# if __name__ == "__main__":
#     datasets = build(128, "/Tmp/data", 1)
