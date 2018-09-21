"""
l2 norm over the entire model
"""
from collections import OrderedDict


def build():
    return l2_norm


def l2_norm(results, name, set_name, loader, model, optimizer, device):
    l2_norm_sum = 0.0
    l2_norm_prod = 1.0
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            squared_w = module.weight * module.weight
            l2_norm_sum += squared_w.sum()
            l2_norm_prod *= squared_w.sum()

    return {'parameters': {'l2_norm': {'sum': l2_norm_sum.item(),
                                       'prod': l2_norm_prod.item()}}}
