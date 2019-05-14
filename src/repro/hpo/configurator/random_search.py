import logging
import pprint
import random

from utils.flatten import unflatten


logger = logging.getLogger(__name__)


def build(space, max_trials, seed):
    return RandomSearch(space, max_trials, seed)


class RandomSearch(object):
    def __init__(self, space, max_trials, seed=1):
        self.space = space
        self.max_trials = max_trials
        self.seed = seed
        self.trials = []

    def is_completed(self):
        return len(self.trials) >= self.max_trials

    def observe(self, params, objective):
        self.trials.append((params, objective))

    def get_params(self, seed=None):
        if seed is None:
            seed = random.randint(0, 100000)
        arguments = unflatten(dict(zip(self.space.keys(), self.space.sample(seed=seed)[0])))
        logger.debug('Sampling:\n{}'.format(pprint.pformat(arguments)))
        return arguments
