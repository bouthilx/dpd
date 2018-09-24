from collections import OrderedDict

from kleio.core.utils import flatten, unflatten

import torch


def build(over):
    return ComputeParticipationRatio(over)


class ComputeParticipationRatio(object):
    def __init__(self, over):
        self.over = over

    def __call__(self, results, name, set_name, analysis_loader, training_loader, model, optimizer, device):
        participation_ratio = OrderedDict()
        flat_results = flatten(results)
        for key, value in flatten(self.over).items():
            key = "{}.{}".format(key, value)
            if key not in flat_results and not any(k.startswith(key) for k in flat_results.keys()):
                raise RuntimeError("Cannot compute participation_ratio; Canot find "
                                   "prior result {}".format(key))

            if key not in flat_results:
                for flat_key in list(flat_results.keys()):
                    if not flat_key.startswith(key):
                        continue
                    subkey = flat_key[len(key) + 1:]
                    eig_key = curate(key + ".participation_ratio" + "." + subkey)
                    participation_ratio[eig_key] = _compute_participation_ratio(
                        flat_results.pop(flat_key))
            else:
                participation_ratio[curate(key + ".participation_ratio")] = _compute_participation_ratio(
                    flat_results.pop(key))

        # flat_results.update(participation_ratio)
        # return unflatten(flat_results)  # participation_ratio)
        return unflatten(participation_ratio)


def curate(key):
    return ".".join([name for name in key.split(".") if not name.startswith("_")])


def _compute_participation_ratio(eigenvalues):
    return ((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum() / eigenvalues.shape[0]).item()


def _participation_ratio(data):

    cov = torch.dot(data, data.T)
    diag = torch.diag(cov)

    return (diag.sum() ** 2) / (diag ** 2).sum() / data.shape[0]
