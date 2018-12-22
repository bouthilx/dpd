import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


# Checkpoints of models pre-trained on Imagenet.
# NOT SUPPORTED YET
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


# Possible configuration variants of VGG network
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, layers, input_size, init_weights, batch_norm, classifier, num_classes):
        super(VGG, self).__init__()
        self.features = self.make_layers(input_size[0], layers, batch_norm)
        
        if classifier.get('hidden'):
            self.classifier = nn.Sequential(
                nn.Linear(classifier['input'], classifier['hidden']),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier['hidden'], classifier['hidden']),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier['hidden'], num_classes),
            )
        else:
            self.classifier = nn.Linear(classifier['input'], num_classes)
            
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def make_layers(self, in_channels, cfg, batch_norm):
        layers = []
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # con2d.register_forward_hook(save_computations)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
        
        
    def save_computations(self, input, output):
        setattr(self, "input", input)
        setattr(self, "output", output)


def distribute(model, distributed):
    # AlexNet and VGG should be treated differently
    #         # DataParallel will divide and allocate batch_size to all available GPUs
    #         if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #             model.features = torch.nn.DataParallel(model.features)
    #             model.cuda()
    #         else:
    #             model = torch.nn.DataParallel(model).cuda()

    # Because last fc layer is big and not suitable for DataParallel
    # Source: https://github.com/pytorch/examples/issues/144

    if distributed > 1:
        if distributed != torch.cuda.device_count():
            raise RuntimeError("{} GPUs are required by the configuration but {} are currently "
                               "made available to the process.".format(
                                   distributed, torch.cuda.device_count()))

        if isinstance(model.features, nn.Sequential):
            model.features = torch.nn.DataParallel(model.features).cuda()
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model
