"""
l2 norm over the entire model
"""
from collections import OrderedDict


def build():
    return l2_norm


def l2_norm(results, name, set_name, loader, model, optimizer, device):
    l2_norm_sum = 0.0
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            squared_w = module.weight * module.weight
            l2_norm_sum += squared_w.sum()

    return OrderedDict((('l2_norm', l2_norm_sum.item()), ))
