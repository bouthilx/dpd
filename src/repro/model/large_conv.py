import torch.nn as nn
from repro.model.vgg import VGG


def build(input_size, num_classes, width, reductions, batch_norm, classifier=None):
    assert len(reductions) == 4
    widths = [width]
    for reduction in reductions:
        widths.append(int(min(max(widths[-1] * reduction, 512), 1024)))
    layers = [widths[0], 'M', widths[1], 'M', widths[2], widths[3], 'M', widths[4], 512, 'M']
    
    if classifier is None:
        classifier = {'input': 512}

    return VGG(layers, input_size=input_size, init_weights=True, batch_norm=batch_norm,
               classifier=classifier, num_classes=num_classes)
