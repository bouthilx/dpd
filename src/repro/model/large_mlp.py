import numpy

from repro.model.mlp import MLP


def build(input_size, num_classes, width, reductions):
    print(reductions)
    assert len(reductions) == 4

    if isinstance(input_size, list):
        input_size = numpy.product(input_size)

    layers = [max(width, 2 * num_classes)]
    for reduction in reductions:
        layers.append(int(max(layers[-1] * reduction, 2 * num_classes)))
    print(layers)
    return MLP(input_size, num_classes, layers=layers, bias=True)
