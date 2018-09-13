from collections import OrderedDict
import copy

import numpy
import torch


MODEL_SIZE_THRESHOLD = 100000


def build():
    return compute_reference


def compute_reference(results, name, set_name, loader, model, optimizer, device):

    with torch.no_grad():
        return _compute_reference(results, name, set_name, loader, model, optimizer, device)


def _compute_reference(results, name, set_name, loader, model, optimizer, device):
    references = _compute_reference_function(loader, model, device)
    references.update(_compute_reference_parameters(model))
    return references


def compute_number_of_parameters(model):
    number_of_parameters = 0
    for parameter in model.parameters():
        number_of_parameters += numpy.product(parameter.size())

    return number_of_parameters


def _compute_reference_parameters(model):
    references = OrderedDict()

    if compute_number_of_parameters(model) <= MODEL_SIZE_THRESHOLD:
        references['all_parameters'] = torch.cat(tuple(m.view(-1) for m in model.parameters()))

    references.update(
        OrderedDict((key, copy.deepcopy(value)) for key, value in model.named_parameters()))

    return OrderedDict(parameters=OrderedDict((('_references', references), )))


def _compute_reference_function(loader, model, device):
    references = []
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        references.append(model(data))

    if references[0].size(0) != references[-1].size(0):
        references.pop(-1)

    # shape is (batch_idx, sample_idx, dimension_idx)
    return OrderedDict(function=OrderedDict((('_references', torch.stack(references)), )))
