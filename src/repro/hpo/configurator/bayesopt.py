import logging
import pprint
import random

try:
    from orion.core.worker.primary_algo import PrimaryAlgo
except ImportError:
    PrimaryAlgo = None

from repro.utils.flatten import flatten, unflatten


logger = logging.getLogger(__name__)


class BayesianOptimizer:
    def __init__(self, space, max_trials, seed, **kwargs):
        self.primary = PrimaryAlgo(space, {'BayesianOptimizer': kwargs})
        self.primary.algorithm.random_state = seed
        self.max_trials = max_trials
        self.trial_count = 0

    @property
    def space(self):
        return self.primary.space

    def is_completed(self):
        return self.trial_count >= self.max_trials

    def get_params(self, seed=None):
        if seed is None:
            seed = random.randint(0, 100000)

        self.primary.algorithm._init_optimizer()
        optimizer = self.primary.algorithm.optimizer
        optimizer.rng.seed(seed)
        # Giving the same seed could be problematic since optimizer.rng and
        # optimizer.base_estimator.rng would be synchronized and sample the same values.
        optimizer.base_estimator_.random_state = optimizer.rng.randint(0, 100000)
        params = unflatten(dict(zip(self.space.keys(), self.primary.suggest()[0])))
        logger.debug('Sampling:\n{}'.format(pprint.pformat(params)))
        return params

    def observe(self, params, objective):

        params = flatten(params)

        params = [[params[param_name] for param_name in self.space.keys()]]
        results = [dict(objective=objective)]

        self.primary.observe(params, results)


if PrimaryAlgo is not None:
    def build(space, max_trials, seed, strategy, n_initial_points, acq_func, alpha, n_restarts_optimizer,
              noise, normalize_y):
        return BayesianOptimizer(
            space, max_trials=max_trials, seed=seed, strategy=strategy, n_initial_points=n_initial_points,
            acq_func=acq_func, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,
            noise=noise, normalize_y=normalize_y)
