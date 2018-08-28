from sgdad.model.vgg import VGG, cfg


def build(input_size, batch_norm, num_classes):
    layers = cfg['vgg16']
    if input_size == [1, 28, 28]:
        layers.pop(-1)
    return VGG(layers, input_size=input_size, init_weights=True, batch_norm=batch_norm,
               num_classes=num_classes)
