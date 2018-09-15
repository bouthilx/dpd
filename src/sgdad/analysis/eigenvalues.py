from collections import OrderedDict

import torch

from kleio.core.utils import flatten, unflatten


def build(over):
    return ComputeEigenValues(over)


class ComputeEigenValues(object):
    def __init__(self, over):
        self.over = over

    def __call__(self, results, name, set_name, loader, model, optimizer, device):
        eigenvalues = OrderedDict()
        flat_results = flatten(results)
        for key, value in flatten(self.over).items():
            key = "{}.{}".format(key, value)
            if key not in flat_results and not any(k.startswith(key) for k in flat_results.keys()):
                import pdb
                pdb.set_trace()
                raise RuntimeError("Cannot compute eigenvalues; Canot find "
                                   "prior result {}".format(key))

            if key not in flat_results:
                for flat_key in flat_results.keys():
                    if not flat_key.startswith(key):
                        continue
                    subkey = flat_key[len(key) + 1:]
                    eig_key = key + "._eigenvalues" + "." + subkey
                    eigenvalues[eig_key] = _compute_eigenvalues(flat_results[flat_key])
            else:
                eigenvalues[key + "._eigenvalues"] = _compute_eigenvalues(flat_results[key])
                # Save backup
                # eigenvalues[key + "._"] = flat_results[key]
                # flat_results[key + "._"] = flat_results.pop(key)

        # flat_results.update(eigenvalues)
        return unflatten(eigenvalues)


def _compute_eigenvalues(data):
    if data.size(1) > data.size(0):
        data = data[:, :data.size(0)]
    diff = data - data.mean(0)
    # diff /= (diff.std(0) + 1e-5).unsqueeze(0)
    # diff /= (diff.std(1) + 1e-5).unsqueeze(1)

    cov = torch.mm(diff.t(), diff)

    eigenvalues, _ = torch.symeig(cov)

    return eigenvalues
