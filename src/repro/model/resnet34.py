from repro.model.resnet import BasicBlock, ResNet


def build(input_size, conv, maxpool, avgpool, num_classes):
     return ResNet(BasicBlock, [3, 4, 6, 3], input_size=input_size, conv=conv, maxpool=maxpool,
                   avgpool=avgpool, num_classes=num_classes)
