"""
Implementation of the spectral norm ball as in 
https://arxiv.org/pdf/1711.01530.pdf
"""
from collections import OrderedDict
import copy
import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def build(n_power_iterations, eps):
    return ComputeSpectralNorm(n_power_iterations, eps)


class ComputeSpectralNorm(object):
    def __init__(self, n_power_iterations, eps):
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def __call__(self, results, name, set_name, loader, model, optimizer, device):

        # Dt() define below eq (3.4)

        with torch.no_grads():

            spectral_norm = self.compute_spectral_norm(loader, model, device)

        return {'spectral_norm': spectral_norm}


    def compute_model_spectral_norm(self, loader, model, device):

        expectation = self.compute_expectation(loader, model, device)

    def compute_expectation(self, loader, model, device):

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            model(data)

            # TODO: In models, save output of each module
            dt_product = 1.0
            for module in model.named_modules():
                dt_product *= self.compute_module_spectral_norm(module.output) ** 2
                
        return spectral_norm

    def compute_module_spectral_norm(self, matrix):

        return 1.0  # TODO
