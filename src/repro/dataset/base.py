import os

import numpy

import torch

from repro.utils.factory import fetch_factories
from repro.dataset.wrapper.base import build_wrapper


factories = fetch_factories('repro.dataset', __file__)


def set_data_path(config):
    if "REPRO_DATA_PATH" not in os.environ:
        print('WARNING: Environment variable REPRO_DATA_PATH is not set. '
              'Data will be downloaded in {}'.format(os.getcwd()))

    config['data_path'] = os.environ.get('REPRO_DATA_PATH', os.getcwd())


def generate_indices(n_points, seeds=None):
    n_valid = int(n_points * 0.15)
    n_test = n_valid
    n_train = n_points - n_valid - n_test

    indices = list(range(n_points))
    if seeds:
        rng = numpy.random.RandomState(seeds[0])
        rng.shuffle(indices)

    trainval_indices = indices[:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    if seeds and len(seeds) > 1:
        rng = numpy.random.RandomState(seeds[1])
        rng.shuffle(trainval_indices)

    train_indices = trainval_indices[:n_train]
    valid_indices = trainval_indices[n_train:]

    return dict(train=train_indices, valid=valid_indices, test=test_indices)


def split_data(dataset, batch_size, seed, num_workers=0):

    indices = generate_indices(len(dataset), seed)

    data_loaders = dict()
    for split_name, split_indices in indices.items():
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        data_loaders[split_name] = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return data_loaders
 
 
def build_dataset(name, batch_size, seed=None, num_workers=0, **kwargs):
    set_data_path(kwargs)
    wrapper = kwargs.pop('wrapper', None)
    data = factories[name](**kwargs)

    splitted_data = split_data(data.pop('dataset'), batch_size=batch_size, seed=seed,
                               num_workers=num_workers)
    data.update(splitted_data)

    if 'input_size' not in data or 'num_classes' not in data:
        # Get one batch
        data['input_size'] = 0
        data['num_classes'] = 0

        # Run over many mini-batches to make sure all classes are seen
        for i, (source, target) in enumerate(data['train']):
            data['input_size'] = list(source.size()[1:])
            data['num_classes'] = max(data['num_classes'], target.max().item() + 1)

            if i > 20:
                break

    if wrapper:
        data = build_wrapper(data, **wrapper)

    return data
