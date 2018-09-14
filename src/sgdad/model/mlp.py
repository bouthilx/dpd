import math

import numpy

import torch
import torch.nn as nn


# MLP with two hidden layers, each with 200 neurons, and ReLU activations.

class MLP(nn.Module):
    def __init__(self, input_size=28*28, num_classes=10, layers=tuple(), bias=True):
        self.architecture = 'MLP--2x200_neurons--input_size=' + str(input_size) + '--num_classes=' + str(num_classes)
        self.input_size = input_size
        super(MLP, self).__init__()
        insizes = [input_size] + list(layers)
        outsizes = list(layers) + [num_classes]
        for i, [insize, outsize] in enumerate(zip(insizes, outsizes)):
            setattr(self, 'fc{}'.format(i), nn.Linear(insize, outsize, bias=bias))

    def forward(self, x):
        x = x.view(x.size(0), self.input_size)
        layers = list(self.named_children())
        for name, layer in layers[:-1]:
            x = nn.functional.relu(layer(x))

        return layers[-1][1](x)


def build(input_size, num_classes, layers, bias):
    if isinstance(input_size, list):
        input_size = numpy.product(input_size)

    return MLP(input_size=input_size, num_classes=num_classes, layers=layers, bias=bias)
