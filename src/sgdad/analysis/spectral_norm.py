"""
Implementation of the spectral norm ball as in 
https://arxiv.org/pdf/1711.01530.pdf
"""
from collections import OrderedDict
import copy
import logging

import torch
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def build(n_power_iterations, eps):
    return ComputeSpectralNorm(n_power_iterations, eps)


class ComputeSpectralNorm(object):
    def __init__(self, n_power_iterations, eps):
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def __call__(self, results, name, set_name, analysis_loader, training_loader, model, optimizer,
                 device):

        # Dt() define below eq (3.4)

        with torch.no_grad():

            spectral_norm = self.compute_model_spectral_norm(analysis_loader, model, device)

        return {'spectral_norm': spectral_norm.item()}


    def compute_model_spectral_norm(self, analysis_loader, model, device):

        expectation = self.compute_expectation(analysis_loader, model, device)

        layers_spectral_norm = 1.0

        for module in model.named_modules():
            if module.__class__.__name__ == 'Linear' or module.__class__.__name__ == 'Conv2d':
                layers_spectral_norm *= compute_module_spectral_norm(module._parameters['weight'])

        return torch.sqrt(expectation) * layers_spectral_norm

    def compute_expectation(self, analysis_loader, model, device):
        expectation = 0.0
        n_samples = 0
        for batch_idx, (data, target) in enumerate(analysis_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            flattened_data = data.view(data.size(0), -1)

            dt_product = (torch.norm(output, 2, 1) != 0).type(output.type())
            expectation += ((torch.norm(flattened_data, 2, 1) ** 2) * dt_product).sum()
            n_samples += data.size(0)
        return expectation / n_samples

    def compute_module_spectral_norm(self, weight_layer):
        n_power_iterations = self.n_power_iterations
        _eps = self.n_power_iterations
        _dim = 0
        weight_height = weight_layer.size(_dim)
        u = normalize(weight_layer.new_empty(weight_height).normal_(0, 1), dim=0, eps=_eps)
        weight_mat = weight_layer
        if _dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(_dim,
                                            *[d for d in range(weight_mat.dim()) if d != _dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=_eps)
                u = normalize(torch.matmul(weight_mat, v), dim=0, eps=_eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        return sigma
