import copy
import math

import numpy

import torch


class ShufflerIterator(object):
    def __init__(self, iterator):
        self.iterator = iterator

    def __next__(self):
        items = next(self.iterator)

        spherical_cows = []

        for item in items:
            if item.type() == "torch.FloatTensor":
                mean = item.mean(0)
                std = item.std(0)
                spherical_cow = torch.randn_like(item) * std + mean
            elif item.type() == "torch.LongTensor":
                lower_bound = item.min()
                upper_bound = item.max()
                if lower_bound.item() == upper_bound.item() == 0:
                    spherical_cow = torch.zeros_like(item)
                else:
                    spherical_cow = torch.randint_like(item, lower_bound, upper_bound)
            else:
                raise TypeError("Don't know spherical cow species for '{}'".format(item.type()))

            spherical_cows.append(spherical_cow)

        return spherical_cows


class Shuffler(object):
    def __init__(self, dataloader, target, level):
        self.dataloader = dataloader
    def __getattr__(self, name):
        if hasattr(self.dataloader, name):
            return getattr(self.dataloader, name)

    def __iter__(self):
        return SphericalCowIterator(iter(self.dataloader))


def shuffle_source(source, level):
    num_examples = source.size(0)
    num_features = numpy.product(source.size()[1:])
    probs = torch.ones(num_features, dtype=torch.float)
    # TODO: round to closest even number
    num_samples = int(math.ceil(num_features * level / 2) * 2)

    original_size = source.size()
    source = source.view(num_examples, -1)
    shuffled_source = torch.zeros(num_examples, num_features)
    for i in range(num_examples):
        indexes = torch.multinomial(probs, num_samples, replacement=False)
        A = indexes[:int(num_samples / 2)]
        B = indexes[int(num_samples / 2):]
        shuffled_source[i] = source[i]
        shuffled_source[i][A] = source[i][B]
        shuffled_source[i][B] = source[i][A]
    # for each multinomial, copy the shuffled version of source in shuffled_source
    #   flatten source
    #   shuffle
    #   copy in data
    return shuffled_source.view(original_size)


def shuffle_target(target, level, num_classes):
    # with probability p, draw a new target
    drop_mask = (torch.rand(target.size()) < level).type(torch.LongTensor)
    random_targets = torch.randint(0, num_classes, target.size()).type(torch.LongTensor)
    shuffled = target * (1 - drop_mask) + random_targets * drop_mask
    return shuffled


def shuffle(loader, target, level, num_classes):
    if target not in ["target", "source"]:
        raise ValueError("Invalid target: {}".format(target))

    # Iterate on all dataset
    sources = []
    targets = []
    for batch_source, batch_target in loader:
        if target == "source":
            batch_source = shuffle_source(batch_source, level)
        elif target == "target":
            batch_target = shuffle_target(batch_target, level, num_classes)

        sources.append(batch_source)
        targets.append(batch_target)

    # Create shuffling for each example
    sources = torch.cat(sources)
    targets = torch.cat(targets)
    shuffled_loader = copy.deepcopy(loader)
    shuffled_loader.dataset = torch.utils.data.TensorDataset(sources, targets)

    return shuffled_loader


def build(data, target, level):
    wrapped_data = {}
    wrapped_data.update(data)
    
    wrapped_data['train'] = shuffle(data['train'], target=target, level=level,
                                    num_classes=data['num_classes'])

    return wrapped_data
