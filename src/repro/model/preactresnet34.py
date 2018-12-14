from repro.model.preactresnet import PreActBlock, PreActResNet


def build(input_size, num_classes):
    return PreActResNet(PreActBlock, [3,4,6,3], input_size=input_size, num_classes=num_classes)
