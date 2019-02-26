import torch.nn as nn
from repro.model.vgg import VGG


def build(input_size, num_classes, widths, batch_norm, classifier=None):
    assert len(widths) == 2
    layers = [widths[0], 'M', widths[1], 'MM', 512, 'MM']
    if classifier is None:
        classifier = {'input': 512}
    return VGG(layers, input_size=input_size, init_weights=True, batch_norm=batch_norm,
               classifier=classifier, num_classes=num_classes)
