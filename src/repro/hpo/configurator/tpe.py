import logging
import pprint
import random

try:
    from orion.core.worker.primary_algo import PrimaryAlgo
    import optuna
except ImportError:
    PrimaryAlgo = None
    optuna = None

from utils.flatten import flatten, unflatten


logger = logging.getLogger(__name__)


class TPEOptimizer:

    def __init__(self, space, max_trials, seed, **kwargs):
        self.primary = PrimaryAlgo(space, {'TPEOptimizer': kwargs})
        self.primary.algorithm.random_state = seed
        self.max_trials = max_trials
        self.trial_count = 0

    @property
    def space(self):
        return self.primary.space

    def is_completed(self):
        return self.trial_count >= self.max_trials

    def get_params(self, seed):

        if seed is None:
            seed = random.randint(0, 100000)

        self.primary.algorithm.study.sampler.rng.seed(seed)
        self.primary.algorithm.study.sampler.random_sampler.rng.seed(seed)

        params = unflatten(dict(zip(self.space.keys(), self.primary.suggest()[0])))
        logger.debug('Sampling:\n{}'.format(pprint.pformat(params)))
        return params

    def observe(self, params, objective):

        params = flatten(params)

        params = [[params[param_name] for param_name in self.space.keys()]]
        results = [dict(objective=objective)]

        self.primary.observe(params, results)


if optuna is not None:
    def build(space, max_trials, seed, **kwargs):
        return TPEOptimizer(space, max_trials=max_trials, seed=seed, **kwargs)
