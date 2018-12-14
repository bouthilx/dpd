from repro.model.preactresnet import PreActBottleneck, PreActResNet


def build(input_size, num_classes):
    return PreActResNet(PreActBottleneck, [3,4,6,3], input_size=input_size, num_classes=num_classes)
