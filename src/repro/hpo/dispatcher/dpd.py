import bisect
import logging

import numpy

import scipy.integrate

from repro.hpo.dispatcher.dispatcher import HPODispatcher


logger = logging.getLogger(__name__)


class DPD(HPODispatcher):
    def __init__(self, space, configurator_config, max_trials, seed,
                 initial_population=200, final_population=20, n_steps=30,
                 window_size=11, n_points=121, min_population=3, population_growth_brake=0.5):

        super(DPD, self).__init__(space, configurator_config, max_trials, seed)

        # initial_population = max_trials
        population_growth_brake = 0.1

        self.initial_population = initial_population
        self.final_population = final_population
        self.n_steps = n_steps
        self.window_size = window_size
        self.padding = int(window_size / 2)
        self.ratio = (final_population / initial_population) ** (1 / n_steps)
        self.percentile = int(self.ratio * 100 + 0.5)
        self.min_population = min_population
        self.population_growth_brake = population_growth_brake

        # TODO: Reimplement epochs_buffer
        self.epochs_buffer = 1
        self.first_epoch = 0

        self.trials = {}

        n_points *= 2  # Add some buffer just in case...

        # self.metrics = numpy.zeros((2000, 300))
        self.metrics = numpy.ones((int(initial_population * 1.1), n_points + 1)) * -1
        self.smooth_metrics = numpy.ones((int(initial_population * 1.1), n_points + 1)) * -1

    def _observe(self, trial_id, params, step, objective, finished=False, **kwargs):

        super(DPD, self)._observe(trial_id, params, step, objective, finished, **kwargs)

        self.add_trial(trial_id)
        self.add_metric(trial_id, step, objective)

    def is_completed(self):
        return sum(self.metrics[:, -1] >= 0) >= self.final_population

    def add_trial(self, trial_id):
        added = trial_id not in self.trials

        if added:
            self.trials[trial_id] = len(self.trials)

        return added

    def add_metric(self, trial_id, step, value):
        # NOTE: We ignore step -1
        if step < 0 or step >= self.metrics.shape[1]:
            return

        metric_key = self.trials[trial_id]
        self.metrics[metric_key][step] = value

        # NOTE: this may fail if some steps are missing in metrics and have default -1
        if step > 1:
            metrics = self.metrics[metric_key][1:step + 1]
            self.smooth_metrics[metric_key][step] = metrics[numpy.where(metrics > 0)].min()
        else:
            self.smooth_metrics[metric_key][step] = self.metrics[metric_key][step]

    def get_decisive_steps(self):
        """Compute at which steps suspension should occur based on metric history"""
        # Compute variance of performances at each epoch across all trials so far.
        mask = (self.metrics >= 0)
        n_points = mask.sum(axis=0)
        # # Ignore epochs where there is less trials than final_population
        mask *= (n_points >= self.final_population)

        # print(var)
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

        while len(steps) < self.n_steps:
            steps.append(steps[-1] + 1)

        return steps

    def should_resume(self, trial_id):
        return not self.should_suspend(trial_id)
        metric_index = self.trials[trial_id]
        trial_metrics = self.smooth_metrics[metric_index]

        epochs = numpy.where(trial_metrics >= 0)[0]
        if epochs.shape[0] == 0:
            return False

        last_epoch = epochs[-1]

        # if last_epoch < 1:
        #     self._resume(trial_id, 'Should not be suspended before epoch 1')
        #     continue
        decisive_epochs = self.get_decisive_steps()

        idx = max(bisect.bisect(decisive_epochs, last_epoch) - 1, 0)
        decisive_epoch = decisive_epochs[idx]

        if last_epoch < decisive_epoch:
            message = 'Should not be suspended before epoch {}. (Was {}) ({})'.format(
                decisive_epoch, last_epoch, decisive_epoch)
            logger.debug(message)
            return True

        percentile, population = self.fetch_results(last_epoch)

        if percentile is None:
            return False

        trial_value = trial_metrics[last_epoch]
        # Deliberately avoid = median to limit unstability
        # (if one trial moves around median it would get suspended and resumed very often)
        population_at_step = self.initial_population * (self.ratio) ** (idx + 1)
        enough_population = population >= int(population_at_step * self.population_growth_brake)
        if trial_value < percentile and enough_population:
            population_message = '{} >= {} ({} * {} * {} ** {})'.format(
                population, population_at_step * self.population_growth_brake,
                self.population_growth_brake, self.initial_population, self.ratio, idx + 1)
            message = 'value({}) < percentile({}) at epoch({}) with population({})'.format(
                trial_value, percentile, decisive_epoch, population_message)
            logger.debug(message)
            return True
        elif trial_value < percentile and not enough_population:
            message = 'value({}) < percentile({}) but population too small: {} < {} ({} * {} * {} ** {})'.format(
                trial_value, percentile,
                population, population_at_step * self.population_growth_brake,
                self.population_growth_brake, self.initial_population, self.ratio, idx + 1)
            logger.debug(message)
            return False

        message = 'value({}) > percentile({}) at epoch({}) with population({})'.format(
            trial_value, percentile, decisive_epoch, population)
        logger.debug(message)

        return False

    def should_suspend(self, trial_id):
        # if epoch < self.grace:
        #     return

        last_step, objective = self.get_objective(trial_id)

        if self.epochs_buffer is None:
            return

        if last_step < self.first_epoch + self.epochs_buffer:
            return

        decisive_epochs = self.get_decisive_steps()

        if last_step < decisive_epochs[0]:
            return

        idx = max(bisect.bisect(decisive_epochs, last_step) - 1, 0)
        decisive_epoch = decisive_epochs[idx]
        # print('decisive_epoch:', decisive_epoch)

        percentile, population = self.fetch_results(decisive_epoch)

        population_at_step = self.initial_population * (self.ratio) ** (idx + 1)
        # print(decisive_epoch, percentile, population, self.final_population, population_at_step)
        # If population is too small but should soon be fine, use percentile to determine
        # suspension
        # Otherwise, suspend until population is large enough.
        # TODO: Limiting progression to having giving lot of time to bad trials at beginning
        #       This brake can be adjusted with self.population_growth_brake
        if population < population_at_step * self.population_growth_brake:
            message = 'Population too small: {} < {} ({} * {} * {} ** {})'.format(
                population, population_at_step * self.population_growth_brake,
                self.population_growth_brake, self.initial_population, self.ratio, idx + 1)
            logger.debug(message)
            return True
        elif percentile is None:
            return False

        decisive_value = self.smooth_metrics[self.trials[trial_id], decisive_epoch]
        # print(decisive_value)
        if decisive_value >= percentile:
            message = 'value({}) > percentile({}) at epoch({}) with population({})'.format(
                decisive_value, percentile, decisive_epoch, population)
            logger.debug(message)
            return True

    def fetch_results(self, epoch, include_task=True):
        mask = self.smooth_metrics[:, epoch] >= 0
        population = mask.sum()
        if population < self.min_population:
            return None, population

        indices = numpy.where(mask)
        # print(self.smooth_metrics[indices, epoch], self.percentile)
        return numpy.percentile(self.smooth_metrics[indices, epoch], self.percentile), population

    def update_epoch_buffer(self, epoch):
        if self.epochs_buffer:
            return

        if self.first_epoch is None:
            self.first_epoch = epoch

        # We assume events we first find are running, since the task is actually executing this part
        # of the code.
        last_event = None
        for event in self.task._status.events[::-1]:
            if event['item']['name'] != 'Running':
                break
            last_event = event

        if last_event is None:
            return

        start_time = last_event['id'].generation_time

        second_epoch = None
        first_epoch = None
        for metric_event in self.task._metrics.events[::-1]:
            # Stop if we passed the current execution time.
            if metric_event['id'].generation_time < start_time:
                break

            # Ignore metrics like usage.
            if metric_event['item']['type'] != 'stat':
                continue

            second_epoch = first_epoch
            first_epoch = metric_event['id'].generation_time

        if second_epoch is None:
            return

        time_per_epoch = second_epoch - first_epoch
        # Removing time_per_epoch because overhead counts time until beginning of first epoch, not
        # end of first epoch.
        overhead = ((first_epoch - start_time) - time_per_epoch).total_seconds()
        time_per_epoch = time_per_epoch.total_seconds()
        self.epochs_buffer = max(int(overhead / time_per_epoch + 0.5), 1)
        print('Estimating overhead and compute time per epochs.')
        print('         Started on:', start_time)
        print(' End of first epoch:', first_epoch)
        print('End of second epoch:', second_epoch)
        print('           Overhead:', overhead)
        print('    Time per epochs:', time_per_epoch)
        print('   Estimated buffer:', self.epochs_buffer)
        print('    First epoch was:', self.first_epoch)
        print('  Won\'t stop before:', self.first_epoch + self.epochs_buffer)

        # started
        # first metric (overhead)
        # second metric (time per epochs)


def build(space, configurator_config, max_trials, seed, **kwargs):
    return DPD(space, configurator_config, max_trials, seed, **kwargs)
