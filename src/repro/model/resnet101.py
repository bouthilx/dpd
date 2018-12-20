import torch

from repro.model.resnet import Bottleneck, ResNet


def build(input_size, conv, maxpool, avgpool, num_classes, distributed=0):
    model = ResNet(Bottleneck, [3, 4, 23, 3], input_size=input_size, conv=conv, maxpool=maxpool,
                   avgpool=avgpool, num_classes=num_classes)

    if distributed > 1:
        if distributed != torch.cuda.device_count():
            raise RuntimeError("{} GPUs are required by the configuration but {} are currently "
                               "made available to the process.".format(
                                   distributed, torch.cuda.device_count()))

        model = torch.nn.DataParallel(model).cuda()

    return model
