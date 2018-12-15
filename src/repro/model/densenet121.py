from repro.model.densenet import Bottleneck, DenseNet


def build(input_size, num_classes):
    return DenseNet(Bottleneck, [6, 12, 24, 16], input_size=input_size, num_classes=num_classes,
                    growth_rate=32)
