import logging
import pprint
import random

from repro.utils.flatten import unflatten


logger = logging.getLogger(__name__)


def build(space, seed):
    return RandomSearch(space, seed)


class RandomSearch(object):
    def __init__(self, space, seed=1):
        self.space = space
        self.seed = seed
        self.trials = []

    def is_completed(self):
        return False

    def observe(self, params, objective):
        self.trials.append((params, objective))

    def get_params(self, seed=None):
        if seed is None:
            seed = random.randint(0, 100000)
        arguments = unflatten(dict(zip(self.space.keys(), self.space.sample(seed=seed)[0])))
        logger.debug('Sampling:\n{}'.format(pprint.pformat(arguments)))
        return arguments
