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

    def __call__(self, results, name, set_name, loader, model, optimizer, device):

        # Dt() define below eq (3.4)

        with torch.no_grads():

            spectral_norm = self.compute_spectral_norm(loader, model, device)

        return {'spectral_norm': spectral_norm}


    def compute_model_spectral_norm(self, loader, model, device):

        # Raise RuntimeError if analysis cannot be executed on such model architecture.
        self.verify_model(model)

        expectation = self.compute_expectation(loader, model, device)

        layers_spectral_norm = 1.0

        for module in model.named_modules():
            layers_spectral_norm *= compute_module_spectral_norm(module._parameters['weight'])

        return torch.sqrt(expectation) * layers_spectral_norm

    def compute_expectation(self, loader, model, device):
        expectation = 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            model(data)
            # TODO: In models, save output of each module
            dt_product = 1.0
            for module in model.named_modules():
                # Since we only use ReLU for non-linearities, we can use the line below
                # Note that d_t is computed from layer activations
                if module.__class__.__name__ == "ReLU":
                    # TODO: from each activation outputs, get the diagonal matrix of its derivatives
                    dt_product *= self.compute_module_spectral_norm(module.output) ** 2

            expectation += (data.norm() ** 2) * dt_product
        return expectation / len(loader)


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

    def verify_model(self, model):

        def is_not_supported(module):
            has_bias = getattr(module, 'bias', None) is not None
            has_conv = module.__class__.__name__ == "Conv2d"
            # Bah, there is nothing else than ReLU in our models, lets be lazy.
            has_non_relu = False
            return has_bias or has_conv or has_non_relu

        for module in model.named_modules():
            if is_not_supported(module):
                raise RuntimeError(TOO_COMPLEX_MODEL_ERROR)
