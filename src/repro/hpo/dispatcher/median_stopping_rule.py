import numpy

from repro.hpo.dispatcher.dispatcher import HPODispatcher


def build(space, configurator_config, max_trials, seed, grace_period, min_samples_required):
    return MedianStoppingRule(space, configurator_config, max_trials, seed, grace_period,
                              min_samples_required)


class MedianStoppingRule(HPODispatcher):

    def __init__(self, space, configurator_config, max_trials, seed, grace_period=60,
                 min_samples_required=3):
        super(MedianStoppingRule, self).__init__(space, configurator_config, max_trials, seed)

        self.grace_period = grace_period
        self.min_samples_required = min_samples_required

    def _compute_median(self, last_step):
        values = []
        population = 0
        for trial_objectives in self.observations.values():
            if last_step in trial_objectives:
                observations = [obs for step, obs in trial_objectives.items() if step <= last_step]
                values.append(numpy.array(observations).mean())
                population += 1

        return numpy.median(values), population

    def should_suspend(self, trial_id: str) -> bool:

        last_step, objective = self.get_objective(trial_id)

        if objective is None or last_step < self.grace_period:
            return False

        median, population = self._compute_median(last_step)

        if population < self.min_samples_required:
            return False

        return objective > median

    def should_resume(self, trial_id) -> bool:
        return False

    def is_completed(self):
        pass

    def _observe(self, trial_id, step, objective, **kwargs):
        self.observations[trial_id][step] = objective
