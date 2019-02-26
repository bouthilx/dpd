import logging
import pprint
import random

from repro.utils.flatten import unflatten


logger = logging.getLogger(__name__)


def build(space, max_trials):
    return RandomSearch(space, max_trials)


class RandomSearch(object):
    def __init__(self, space, max_trials):
        self.space = space
        self.max_trials = max_trials
        self.trials = []

    def get_bests(self, max_resource):
        # Include non completed trials to avoid ignoring them.
        best_trials = [trial for trial in self.trials
                       if trial['registry']['status'] != 'Completed']

        def validation_error(trial):
            try:
                return trial['output']['best']['valid']['error_rate']
            except KeyError:
                pprint.pprint(trial)
                raise

        completed_trials = (trial for trial in self.trials
                            if trial['registry']['status'] == 'Completed' and trial['output'])

        best_trials += list(sorted(completed_trials, key=validation_error))

        return best_trials[:max_resource]

    def is_completed(self):
        return len(self.trials) >= self.max_trials

    def observe(self, trials):
        self.trials += trials

    def get_params(self, seed=None):
        if seed is None:
            seed = random.randint(0, 100000)
        arguments = unflatten(dict(zip(self.space.keys(), self.space.sample(seed=seed)[0])))
        logger.debug('Sampling:\n{}'.format(pprint.pformat(arguments)))
        return arguments
