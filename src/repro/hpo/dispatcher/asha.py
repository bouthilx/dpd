from collections import defaultdict
import argparse
import logging
import pprint
import random

from typing import Dict

from repro.utils.flatten import flatten, unflatten
from repro.hpo.dispatcher.dispatcher import HPODispatcher


logger = logging.getLogger(__name__)


# # NOTE: task name automatically logged in tags
# mahler.register(task, after=None, before=None, tags=())
# mahler.task_count(tags, status=None)
# mahler.find(tags, status=None)


def build(space, configurator_config, fidelities, reduction_factor, max_resource, max_trials, seed):
    return ASHA(space, configurator_config, fidelities, reduction_factor, max_resource, max_trials, seed)


class ASHA(HPODispatcher):
    def __init__(self, space, configurator_config, fidelities, reduction_factor, max_resource, max_trials, seed):
        super(ASHA, self).__init__(space, configurator_config, max_trials, seed)

        self.fidelities = fidelities
        self.rungs = defaultdict(set)
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.base = len(self.fidelities) - 1

    def _observe(self, trial_id, params, step, objective, finished=False, **kwargs):
        if trial_id not in self.observations:
            rung_id = 0
            self.rungs[0].add(trial_id)

        super(ASHA, self)._observe(trial_id, params, step, objective, finished, **kwargs)

    def make_suggest_parameters(self) -> Dict[str, any]:
        kwargs = super(ASHA, self).make_suggest_parameters()

        self.rungs[0].add(kwargs['trial_id'])
        self.update_rungs()

        return kwargs

    def is_completed(self):
        completed = 0
        for trial_id in self.rungs.get(self.base, []):
            last_step, _ = self.get_objective(trial_id)
            completed += int(last_step >= self.fidelities[-1])

            if completed >= self.max_resource:
                return True

        return False

    def get_rung_id(self, trial_id):
        for rung_id, rung in sorted(self.rungs.items(), key=lambda item: item[0], reverse=True): 
            if trial_id in rung:
                return rung_id

        return None

    def should_suspend(self, trial_id):
        rung_id = self.get_rung_id(trial_id)
        last_step = max(self.observations[trial_id].keys())
        return self.fidelities[rung_id] <= last_step

    def should_resume(self, trial_id):
        return not self.should_suspend(trial_id)

    def get_candidate(self, rung_id):
        rung = self.rungs.get(rung_id, set())
        next_rung = self.rungs.get(rung_id + 1, set())
        budget = self.fidelities[rung_id]
        k = len(rung) // self.reduction_factor

        completed_trials = []
        for trial_id in rung:
            last_step, objective = self.get_objective(trial_id)
            if last_step and last_step >= budget:
                completed_trials.append((objective, trial_id))

        completed_trials = list(sorted(completed_trials))
        i = 0
        k = min(k, len(completed_trials))
        while i < k:
            objective, trial_id = completed_trials[i]
            if trial_id not in next_rung:
                return trial_id, objective
            i += 1

        return None, None

    def update_rungs(self):
        """

        Notes
        -----
            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.
        """
        # NOTE: There should be base + 1 rungs
        for rung_id in range(self.base - 1, -1, -1):
            candidate, objective = self.get_candidate(rung_id)

            if candidate:
                logger.info(f'{candidate} promoting to {rung_id + 1} {objective}')
                self.rungs[rung_id + 1].add(candidate)
                logger.info(
                    'Current rungs: ' +
                    ' '.join('{}:{}'.format(rung_id, len(rung))
                             for rung_id, rung in self.rungs.items()))
                return


# asha = ASHA(rung, max_resource=256, min_resource=1, reduction_factor=4,
#             minimum_early_stopping_rate=0)

def test():
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument(
        '--reduction-factor', type=int, required=True,
        help='Reduction factor for number of workers')
    parser.add_argument(
        '--max-resource', default=1, type=int, help='Number of trials to run at full fidelity')
    parser.add_argument(
        '--num-workers', type=int, help='Number of workers')

    options = parser.parse_args()

    if options.num_workers is None:
        options.num_workers = options.max_resource

    import mahler.core.status

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
