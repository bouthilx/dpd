from repro.model.resnet import Bottleneck, ResNet


def build(input_size, conv, maxpool, avgpool, num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], input_size=input_size, conv=conv, maxpool=maxpool,
                  avgpool=avgpool, num_classes=num_classes)
