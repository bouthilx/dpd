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
    return


# Compute the expectation on the entire set...


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

        # Dt() define below eq (3.4)

        with torch.no_grads():
            fisher_rao_norms = self.compute_fisher_rao_norm(loader, model, device)

        return fisher_rao_norms

    def compute_fisher_rao_norm(self, loader, model, device):

        expectation = 0.0
        emp_expectation = 0.0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            softmax = torch.nn.softmax(output)

            torch.dot(softmax, output) - softmax

        return spectral_norm

    def compute_module_spectral_norm(self, matrix):

        return 1.0  # TODO

    def verify_model(self, model):
        for module in model.named_modules():
            if getattr(module, 'bias', None) is not None:
                raise RuntimeError(TOO_COMPLEX_MODEL_ERROR)
