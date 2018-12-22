from repro.model.vgg import VGG, distribute, cfg


def build(input_size, batch_norm, classifier, num_classes, distributed=0):
    model = VGG(cfg['vgg16'], input_size=input_size, init_weights=True, batch_norm=batch_norm,
                classifier=classifier, num_classes=num_classes)

    return distribute(model, distributed)
