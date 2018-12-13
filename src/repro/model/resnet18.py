from repro.model.resnet import BasicBlock, ResNet


def build(input_size, conv, maxpool, avgpool, num_classes):
     return ResNet(BasicBlock, [2, 2, 2, 2], input_size=input_size, conv=conv, maxpool=maxpool,
                   avgpool=avgpool, num_classes=num_classes)
