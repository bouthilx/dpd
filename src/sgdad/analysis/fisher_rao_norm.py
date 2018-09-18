from collections import OrderedDict
import torch
import torch.nn.functional as F

# https://arxiv.org/pdf/1711.01530.pdf
# Eq. (3.2)
# Eq. (5.1) for classifier

# Diff between model and empirical
# Eq (5.2) model  (average over y)
# Eq (5.3) empirical
# Notes:
#    g() is the softmax()
#    L is the number of layers
#    m is the size of the test set


def build():
    return ComputeFisherRaoNorm()


TOO_COMPLEX_MODEL_ERROR = """\
Simplified Fisher-Rao norm can only be computed on fully connected layers with ReLU activations and
no biases
"""


class ComputeFisherRaoNorm(object):
    def __init__(self):
        pass

    def __call__(self, results, name, set_name, loader, model, optimizer, device):

        with torch.no_grad():
            fisher_rao_norms = self.compute_fisher_rao_norm(loader, model, device)

        return fisher_rao_norms

    def compute_fisher_rao_norm(self, loader, model, device):

        # Raise RuntimeError if analysis cannot be executed on such model architecture.
        self.verify_model(model)

        expectation = 0.0
        emp_expectation = 0.0
        number_of_samples = 0.0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            softmax = F.softmax(output)

            expectation += self.true_fisher_rao_norm(output, softmax)
            emp_expectation += self.empirical_fisher_rao_norm(output, softmax, target)
            number_of_samples += data.size(0)

        # This only works for our specific implementation of MLP
        # If there is nested modules the number of layers will be artificially high.
        number_of_layers = sum(1 for _ in model.modules())
        coefficient = (number_of_layers + 1) ** 2 / number_of_samples

        return OrderedDict((('fisher_rao_norm.true', (coefficient * expectation).item()),
                            ('fisher_rao_norm.empirical', (coefficient * emp_expectation).item())))

    def true_fisher_rao_norm(self, output, softmax):
        # (b,m)-dim
        diff = (softmax * output).sum(1).unsqueeze(1) - output
        return (softmax * (diff * diff)).sum()

    def empirical_fisher_rao_norm(self, output, softmax, target):
        # (b)-dim
        diff = (softmax * output).sum(1) - output[torch.arange(output.size(0)), target]
        return (diff * diff).sum()

    def verify_model(self, model):

        def is_not_supported(module):
            has_bias = getattr(module, 'bias', None) is not None
            has_conv = module.__class__.__name__ == "Conv2d"
            # Bah, there is nothing else than ReLU in our models, lets be lazy.
            has_non_relu = False
            return has_bias or has_conv or has_non_relu

        for name, module in model.named_modules():
            if is_not_supported(module):
                raise RuntimeError(TOO_COMPLEX_MODEL_ERROR)
