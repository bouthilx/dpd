import numpy

import torchvision.models as models


def compute_number_of_parameters(model):
    number_of_parameters = 0
    for parameter in model.parameters():
        number_of_parameters += numpy.product(parameter.size())

    return number_of_parameters

model_names = [
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152"]

for model_name in model_names:
    print(model_name, compute_number_of_parameters(getattr(models, model_name)()))


"""
model    # params   # top-1 error 
                      on ImageNet
vgg11    132863336  30.98
vgg13    133047848  30.07
vgg16    138357544  28.41
vgg19    143667240  27.62
vgg11_bn 132868840  29.62
vgg13_bn 133053736  28.45
vgg16_bn 138365992  26.63
vgg19_bn 143678248  25.76
resnet18  11689512  30.24
resnet34  21797672  26.70
resnet50  25557032  23.85
resnet101 44549160  22.63
resnet152 60192808  21.69
"""


"""
model    # params   # top-1 error 
                      on ImageNet
vgg11    132863336  30.98
vgg19    143667240  27.62
vgg11_bn 132868840  29.62
vgg19_bn 143678248  25.76
resnet18  11689512  30.24
resnet50  25557032  23.85
"""
