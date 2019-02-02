from collections import defaultdict
from math import exp, log
import argparse
import logging
import pprint
import random

import mahler.core.status
from mahler.core.utils.flatten import flatten, unflatten


logger = logging.getLogger(__name__)


# # NOTE: task name automatically logged in tags
# mahler.register(task, after=None, before=None, tags=())
# mahler.task_count(tags, status=None)
# mahler.find(tags, status=None)


def build(space, fidelity_space, reduction_factor, max_resource):
    return ASHA(space, fidelity_space, reduction_factor, max_resource)


class ASHA(object):
    def __init__(self, space, fidelity_space, reduction_factor, max_resource):

        self.space = space
        self.fidelity_dim, self.fidelity_levels = next(iter(fidelity_space.items()))
        self.rungs = defaultdict(list)
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.base = len(self.fidelity_levels) - 1

    def reset(self):
        self.rungs = defaultdict(list)

    def _fetch_rung_id(self, trial):
        return self.fidelity_levels.index(trial['arguments'][self.fidelity_dim])

    def observe(self, trials):
        for trial in trials:
            self.rungs[self._fetch_rung_id(trial)].append(trial)

    def final_rung_is_filled(self):
        return len(self.rungs.get(self.base, [])) >= self.max_resource

    def _top_k(self, trials, k):
        completed_trials = (trial for trial in trials
                            if trial['registry']['status'] == 'Completed' and trial['output'])
        # completed_trials = trials

        def key(trial):
            try:
                error_rate = trial['output']['best']['valid']['error_rate']
            except KeyError:
                pprint.pprint(trial)
                raise

            return error_rate

        return [trial for i, trial in enumerate(sorted(completed_trials, key=key)) if i < k]

    def _fetch_trial_params(self, arguments):
        flattened_arguments = flatten(arguments)
        return unflatten(dict((key, flattened_arguments[key]) for key in self.space.keys()))

    def get_params(self):
        """

        Notes
        -----
            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.
        """
        # NOTE: There should be base + 1 rungs
        for k in range(self.base - 1, -1, -1):
            rungs_k = self.rungs.get(k, [])
            candidates = self._top_k(rungs_k, k=len(rungs_k) // self.reduction_factor)

            # Compare based on arguments
            rungs_kp1 = [self._fetch_trial_params(trial['arguments'])
                         for trial in self.rungs.get(k + 1, [])]
            candidates = [candidate for candidate in candidates
                          if self._fetch_trial_params(candidate['arguments']) not in rungs_kp1]

            if candidates:
                arguments = self._fetch_trial_params(candidates[0]['arguments'])
                arguments[self.fidelity_dim] = self.fidelity_levels[k + 1]
                logger.info(
                    'Promoting to {}:\n{}'.format(
                        k + 1, pprint.pformat(arguments)))
                return arguments

        randomseed = random.randint(0, 100000)
        arguments = unflatten(dict(zip(self.space.keys(), self.space.sample(seed=randomseed)[0])))
        arguments[self.fidelity_dim] = self.fidelity_levels[0]
        logger.info('Sampling:\n{}'.format(pprint.pformat(arguments)))
        return arguments


# asha = ASHA(rung, max_resource=256, min_resource=1, reduction_factor=4,
#             minimum_early_stopping_rate=0)

def test():
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument(
        '--reduction-factor', type=int, required=True, help='Reduction factor for number of workers')
    parser.add_argument(
        '--max-resource', default=1, type=int, help='Number of trials to run at full fidelity')
    parser.add_argument(
        '--num-workers', type=int, help='Number of workers')

    options = parser.parse_args()

    if options.num_workers is None:
        options.num_workers = options.max_resource

    class Space(object):
        def sample(self, seed=0):
            return [[random.uniform(0, 1)]]
        def keys(self):
            return ['a']

    class Task(object):
        def __init__(self, arguments, output):
            self.arguments = arguments
            self.output = output
            self.status = mahler.core.status.Running('')

    def create_task(arguments, output):
        return dict(arguments=arguments, output=output, registry=dict(status='Running'))

    fidelities = dict(fidelity='abcd')  # efg')

    space = Space()

    asha = ASHA(
        space, fidelities,
        reduction_factor=options.reduction_factor,
        max_resource=options.max_resource)

    while not asha.final_rung_is_filled():
        tasks = []

        for i in range(options.num_workers):
            if asha.final_rung_is_filled():
                break
            params = asha.get_params()
            task = create_task(
                arguments=params,
                output=dict(best=dict(valid=dict(error_rate=random.uniform(0, 1)))))
            tasks.append(task)
            asha.observe([task])

        for task in tasks:
            task['registry']['status'] = 'Completed'

        print(" ".join("{}:{}".format(next(iter(fidelities.values()))[key], len(asha.rungs[key]))
                       for key in sorted(asha.rungs.keys())))

if __name__ == "__main__":
    test()
