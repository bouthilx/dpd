import numpy

from repro.model.mlp import MLP


def build(input_size, num_classes):
    if isinstance(input_size, list):
        input_size = numpy.product(input_size)

    return MLP(input_size, num_classes, layers=[], bias=True)
