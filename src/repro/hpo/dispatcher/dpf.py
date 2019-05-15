import logging

import numpy

import scipy.integrate

from repro.hpo.dispatcher.dispatcher import HPODispatcher


logger = logging.getLogger(__name__)


class DPF(HPODispatcher):
    def __init__(self, space, configurator_config, max_trials, seed,
                 steps_ratio=0.5, asynchronicity=0.5,
                 initial_population=20, final_population=1,
                 window_size=11, max_epochs=10, min_population=3):

        super(DPF, self).__init__(space, configurator_config, max_trials, seed)

        # Hints for start
        self._initial_population = initial_population
        self._max_epochs = max_epochs

        # Static values
        self._steps_ratio = steps_ratio
        self.final_population = final_population
        self.window_size = window_size
        self.padding = int(window_size / 2)
        self.asynchronicity = asynchronicity

        self.metrics = numpy.ones((int(initial_population * 2), max_epochs + 1)) * -1
        self.smooth_metrics = numpy.ones((int(initial_population * 2), max_epochs + 1)) * -1
        self.suspension_matrix = numpy.ones((int(initial_population * 2), max_epochs + 1))

        self.trials = {}

    def increase_metrics(self, axis):
        for name in ['metrics', 'smooth_metrics']:
            setattr(self, name, self._increase_metrics(getattr(self, name), axis))

    def _increase_metrics(self, metrics, axis):
        shape = metrics.shape
        new_metrics = numpy.ones((shape[0] * (2 if axis == 0 else 1),
                                  shape[1] + (10 if axis == 1 else 0))) * -1
        new_metrics[:shape[0], :shape[1]] = metrics
        return new_metrics

    def _observe(self, trial_id, params, step, objective, finished=False, **kwargs):

        super(DPF, self)._observe(trial_id, params, step, objective, finished, **kwargs)

        self.add_trial(trial_id)
        self.add_metric(trial_id, step, objective)
        self.suspension_matrix = self.compute_suspension_matrix(self.get_decision_steps())

    def is_completed(self):
        return self.finished

    def add_trial(self, trial_id):
        added = trial_id not in self.trials

        if added:
            self.trials[trial_id] = len(self.trials)

        return added

    def add_metric(self, trial_id, step, value):
        metric_key = self.trials[trial_id]

        if metric_key >= self.metrics.shape[0]:
            self.increase_metrics(0)
        elif step >= self.metrics.shape[1]:
            self.increase_metrics(1)

        if step < 0:
            return

        self.metrics[metric_key][step] = value

        # NOTE: this may fail if some steps are missing in metrics and have default -1
        if step > 1:
            metrics = self.metrics[metric_key][1:step + 1]
            self.smooth_metrics[metric_key][step] = metrics[numpy.where(metrics > 0)].min()
        else:
            self.smooth_metrics[metric_key][step] = self.metrics[metric_key][step]

    @property
    def trials_count(self):
        return len(self.trials)

    @property
    def population(self):
        return max(self._initial_population, len(self.trials))

    @property
    def max_epochs(self):
        observed = (self.metrics >= 0).sum(1).max()
        return max(self._max_epochs, observed)

    @property
    def n_steps(self):
        return int(self._steps_ratio * self.max_epochs + 0.5)

    @property
    def ratio(self):
        return (self.final_population / self.population) ** (1 / self.n_steps)

    @property
    def percentile(self):
        return int(self.ratio * 100 + 0.5)

    def get_decision_steps(self):
        steps = self._get_decision_steps()
        # Create a mask based on steps
        suspension = self.compute_suspension_matrix(steps)

        mask = (self.metrics >= 0)
        for i, step in enumerate(steps):
            mask[:self.trials_count, step:] *= (suspension[:, i] < 1)[:, None]

        return self._get_decision_steps(mask)

    def _get_decision_steps(self, mask=None):
        """Compute at which steps suspension should occur based on metric history"""
        # Compute variance of performances at each epoch across all trials so far.
        if mask is None:
            mask = (self.metrics >= 0)

        n_points = mask.sum(axis=0)
        # # Ignore epochs where there is less trials than final_population
        mask *= (n_points >= self.final_population)

        max_metrics = numpy.max(self.smooth_metrics * mask, axis=0)
        min_metrics = numpy.min(self.smooth_metrics * mask + (1 - mask) * max_metrics, axis=0)

        diff = max_metrics - min_metrics

        steps = numpy.where(diff > 0)[0]
        if steps.shape[0] > 0:
            first_step = steps[0]
        else:
            return list(range(2, self.n_steps + 2))

        # Compute the cummulative integral of variance over epochs
        # and pick a list of epochs so that all intervals contain the same
        # total amount of variance.
        dcum = scipy.integrate.cumtrapz(diff[first_step:], initial=0)
        block = dcum[-1] / (self.n_steps - first_step)
        steps = []
        j = 1
        for i, accum in enumerate(dcum):
            if accum > block * j:
                j += 1
                steps.append(i + first_step)

        if not steps:
            steps.append(1)

        while len(steps) < self.n_steps and steps[-1] + 1 < self.max_epochs:
            steps.append(steps[-1] + 1)

        return steps

    def compute_suspension_matrix(self, steps):

        suspend = numpy.zeros((self.trials_count, len(steps)))
        populations = []
        pop = []
        for i, step in enumerate(steps):
            mask = self.smooth_metrics[:, step] >= 0
            reached_step = numpy.where(mask)[0]
            population = self.get_population(step, mask)
            populations.append([i, step, population, self.get_min_population(i)])
            if population:
                pop.append((step, population))
            if self.get_population(step, mask) < self.get_min_population(i):
                suspend[reached_step, i] = 1
                continue

            values = self.smooth_metrics[reached_step, step]
            if len(values) < 2:
                continue

            percentile = numpy.percentile(values, self.percentile)

            populations[-1].append(percentile)

            suspend[reached_step, i:] = (values >= percentile)[:, None]

        return suspend

    def get_min_population(self, ith_step):
        return self.population * (self.ratio) ** (ith_step + 1) * (1 - self.asynchronicity)

    def get_population(self, step, mask):
        return mask.sum()

    def should_suspend(self, trial_id):
        row_id = self.trials[trial_id]
        return bool(self.suspension_matrix[row_id].sum() > 0)

    def should_resume(self, trial_id):
        suspend = self.should_suspend(trial_id)
        last_step, obj = self.get_objective(trial_id)
        return not suspend


def build(space, configurator_config, max_trials, seed, **kwargs):
    return DPF(space, configurator_config, max_trials, seed, **kwargs)
