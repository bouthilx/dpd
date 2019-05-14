import itertools
import logging
import pprint

import numpy

from orion.core.worker.transformer import build_required_space
from utils.flatten import unflatten


logger = logging.getLogger(__name__)


def build(space, max_trials, seed):
    return GridSearch(space, max_trials, seed)


class GridSearch(object):
    def __init__(self, space, max_trials, seed=1):
        self.space = space
        self.linear_space = build_required_space('linear', space)
        self.max_trials = max_trials
        self.trials = []

        n_per_dim = numpy.ceil(numpy.exp(numpy.log(max_trials) / len(space)))
        dimensions = []
        for name, dim in self.linear_space.items():
            assert not dim.shape or len(dim.shape) == 1 and dim.shape[0] == 1
            low, high = dim.interval()
            dimensions.append(list(numpy.linspace(low, high, n_per_dim)))

        self.params = list(itertools.product(*dimensions))

        if len(self.params) > max_trials:
            logger.warning(
                f'GridSearch has more trials ({len(self.params)}) than the max ({max_trials})')

    def is_completed(self):
        return len(self.trials) >= self.max_trials

    def observe(self, params, objective):
        self.trials.append((params, objective))

    def get_params(self, seed=None):
        params = self.linear_space.reverse(self.params[len(self.trials)])
        arguments = unflatten(dict(zip(self.space.keys(), params)))
        logger.debug('Sampling:\n{}'.format(pprint.pformat(arguments)))
        return arguments


if __name__ == '__main__':
    from repro.benchmark.visiondl import build_space
    c = GridSearch(build_space('vgg11', 'sgd'), 100, None)
    print(c.get_params(0))
    c.observe(None, None)
    print(c.get_params(0))
