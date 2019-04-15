import bisect
import pprint
import time

import scipy.integrate

from bson import objectid

import numpy

try:
    import mahler.client as mahler
    from mahler.core.utils.errors import SignalSuspend, RaceCondition
except ImportError:
    mahler = None
    SignalSuspend = None


def removeOutliers(x, outlierConstant):
    a = numpy.array(x)
    upper_quartile = numpy.percentile(a, 75, axis=1)
    lower_quartile = numpy.percentile(a, 0, axis=1)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    import pdb
    pdb.set_trace()
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList


class AsynchronousFilteringPercentile:

    def __init__(self, initial_population, final_population, n_steps=30, window_size=11,
                 n_points=200, min_population=3, population_growth_brake=1):

        self.initial_population = initial_population
        self.final_population = final_population
        self.n_steps = n_steps
        self.window_size = window_size
        self.padding = int(window_size / 2)
        self.ratio = (final_population / initial_population) ** (1 / n_steps)
        self.percentile = int(self.ratio * 100 + 0.5)
        print('Percentile:', self.percentile)
        self.min_population = min_population
        self.population_growth_brake = population_growth_brake

        self.trials = {}
        self.first_epoch = None
        self.epochs_buffer = None
         
        # self.metrics = numpy.zeros((2000, 300))
        self.metrics = numpy.ones((int(initial_population * 1.1), n_points + 1)) * -1
        self.smooth_metrics = numpy.ones((int(initial_population * 1.1), n_points + 1)) * -1

        self.task_timestamp = None
        self.metric_timestamp = None

        self.task = None
        if mahler is not None:
            task_id = mahler.get_current_task_id()
            if task_id is not None:
                self.mahler_client = mahler.Client()
                self.task = list(self.mahler_client.find(id=task_id))[0]
                self.tags = self.task.tags

        if self.task is None:
            print('WARNING: Cannot use MedianDistanceStoppingRule without mahler.')

    def get_decisive_steps(self):
        """Compute at which steps suspension should occur based on metric history"""
        # Compute variance of performances at each epoch across all trials so far.
        mask = (self.metrics >= 0)
        n_points = mask.sum(axis=0)
        # # Ignore epochs where there is less trials than final_population
        mask *= (n_points >= self.final_population)

        # print(var)
        diff = (numpy.max(self.smooth_metrics * mask, axis=0) -
                numpy.min(self.smooth_metrics * mask, axis=0))

        # Compute the cummulative integral of variance over epochs
        # and pick a list of epochs so that all intervals contain the same
        # total amount of variance.
        dcum = scipy.integrate.cumtrapz(diff, initial=0)
        block = dcum[-1] / self.n_steps
        steps = []
        j = 1
        for i, accum in enumerate(dcum):
            if accum > block * j:
                j += 1
                steps.append(i)

        if not steps:
            steps.append(1)

        while len(steps) < self.n_steps:
            steps.append(steps[-1] + 1)

        return steps

    @property
    def db_client(self):
        return self.mahler_client.registrar._db._db

    def close(self):
        self.mahler_client.close()
        
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
        #

    def update(self, stats):
        if self.task is None:
            pprint.pprint(stats)
            return

        self.mahler_client.add_metric(self.task.id, stats, force=True)

        # This will refresh the trials and metrics at the same time.
        self.thaw(subset=10)

        epoch = stats['epoch']
        self.update_epoch_buffer(epoch)

        if self.epochs_buffer is None:
            return

        if epoch < self.first_epoch + self.epochs_buffer:
            return

        # if epoch < self.grace:
        #     return
        print("Should I skip?")
        print(epoch)
        decisive_epochs = self.get_decisive_steps()
        print(decisive_epochs)
        if epoch < decisive_epochs[0]:
            return

        idx = max(bisect.bisect(decisive_epochs, epoch) - 1, 0)
        decisive_epoch = decisive_epochs[idx]
        print('decisive_epoch:', decisive_epoch)

        percentile, population = self.fetch_results(decisive_epoch)

        population_at_step = self.initial_population * (self.ratio) ** (idx + 1)
        print(decisive_epoch, percentile, population, self.final_population, population_at_step)
        # If population is too small but should soon be fine, use percentile to determine
        # suspension
        # Otherwise, suspend until population is large enough.
        # TODO: Limiting progression to having giving lot of time to bad trials at beginning
        #       This brake can be adjusted with self.population_growth_brake
        if population < population_at_step * self.population_growth_brake: #  and not population_is_soon_enough:
            message = 'Population too small: {} < {} ({} * {} * {} ** {})'.format(
                population, population_at_step * self.population_growth_brake, 
                self.population_growth_brake, self.initial_population, self.ratio, idx + 1)
            raise SignalSuspend(message)
        elif percentile is None:
            return

        print(self.metrics[self.trials[str(self.task.id)], decisive_epoch - 3:decisive_epoch + 4])
        print(self.smooth_metrics[self.trials[str(self.task.id)], decisive_epoch - 3:decisive_epoch + 4])
        decisive_value = self.smooth_metrics[self.trials[str(self.task.id)], decisive_epoch]
        print(decisive_value)
        if decisive_value > percentile:
            message = 'value({}) > percentile({}) at epoch({}) with population({})'.format(
                decisive_value, percentile, decisive_epoch, population)
            raise SignalSuspend(message)

    def refresh(self):
        self._update_trials()
        self._update_metrics()

    def _update_trials(self):
        # fetch trials report based on updated timestamp
        query = {'registry.tags': {'$all': self.tags}}

        if self.task_timestamp:
            query['registry.reported_on'] = {'$gt': objectid.ObjectId(self.task_timestamp)}

        projection = {'_id': 1, 'registry.reported_on': 1}

        start = time.time()

        for trial in self.db_client.tasks.report.find(query, projection):
            trial_id = str(trial['_id'])
            if trial_id in self.trials:
                continue

            self.trials[trial_id] = len(self.trials)
            self.task_timestamp = trial['registry']['reported_on']
            # If a trial is added, maybe some metrics were loaded but discarded because trial was
            # not loaded yet. To be safe, fallback metric_timestamp
            if self.metric_timestamp:
                self.metric_timestamp = min(self.task_timestamp, self.metric_timestamp)

        print('trial time:', time.time() - start)

    def _update_metrics(self):
        query = {'item.type': 'stat'}

        if self.metric_timestamp:
            query['_id'] = {'$gt': self.metric_timestamp}

        projection = {'task_id': 1, 'item.value.epoch': 1, 'item.value.valid.error_rate': 1}
        start = time.time()

        # fetch metrics after timestamp, ignore metric if not in trial ids
        for metric in self.db_client.tasks.metrics.find(query, projection):
            task_id = str(metric['task_id'])
            if task_id not in self.trials:
                continue

            stats = metric['item']['value']
            metric_key = self.trials[task_id]
            epoch = stats['epoch']

            self.metrics[metric_key][epoch] = stats['valid']['error_rate']
            # min_values = numpy.minimum.accumulate(self.metrics[metric_key][epoch:])
            # self.metrics[metric_key][epoch:] = min_values

            # +1 to compensate for missing epoch 0
            n_points = (self.metrics[metric_key] >= 0).sum() + 1
            a = max(epoch - self.padding, 1)
            effective_padding = epoch - a
            b = min(epoch + effective_padding + 1, n_points)

            # TODO: Compute all this afterwards using vector computations in numpy
            for o_epoch in range(a, b):
                a = max(o_epoch - self.padding, 1)
                effective_padding = o_epoch - a
                b = min(o_epoch + effective_padding + 1, n_points)

                self.smooth_metrics[metric_key][o_epoch] = self.metrics[metric_key][a:b].mean()

            self.metric_timestamp = metric['_id']

        print('metric time', time.time() - start)

        # print(self.trials)
        # print(self.metrics)
        # print(self.best)

    def fetch_results(self, epoch, include_task=True):
        mask = self.smooth_metrics[:, epoch] >= 0
        population = mask.sum()
        if population < self.min_population:
            return None, population

        indices = numpy.where(mask)
        return numpy.percentile(self.smooth_metrics[indices, epoch], self.percentile), population

    def thaw(self, subset=None):
        if self.task is None:
            return

        self.refresh()
        decisive_epochs = self.get_decisive_steps()
        percentiles = dict()
        if subset and len(self.trials) > subset:
            trial_ids = numpy.random.choice(
                list(self.trials.keys()), size=subset, replace=False)
        else:
            trial_ids = self.trials.keys()

        # TODO: Compute all based on matrix directly and back-track to trial ids.
        for trial_id in trial_ids:
            metric_index = self.trials[trial_id]
            if trial_id == str(self.task.id):
                continue

            trial_metrics = self.smooth_metrics[metric_index]
            # We can ignore epoch 0 because it is set to -1
            last_epoch = sum(trial_metrics >= 0)

            # if last_epoch < 1:
            #     self._resume(trial_id, 'Should not be suspended before epoch 1')
            #     continue
    
            idx = max(bisect.bisect(decisive_epochs, last_epoch) - 1, 0)
            decisive_epoch = decisive_epochs[idx]

            if last_epoch < decisive_epoch:
                message = 'Should not be suspended before epoch {}. (Was {}) ({})'.format(
                    decisive_epoch, last_epoch, str(decisive_epochs))
                self._resume(trial_id, message)
                continue

            if decisive_epoch not in percentiles:
                percentiles[decisive_epoch] = self.fetch_results(decisive_epoch)

            if percentiles[decisive_epoch][0] is None:
                continue

            trial_value = trial_metrics[decisive_epoch]
            percentile, population = percentiles[decisive_epoch]
            # Deliberately avoid = median to limit unstability
            # (if one trial moves around median it would get suspended and resumed very often)
            population_at_step = self.initial_population * (self.ratio) ** (idx + 1)
            enough_population = population >= population_at_step * self.population_growth_brake
            if trial_value < percentile and enough_population:
                population_message = '{} >= {} ({} * {} * {} ** {})'.format(
                    population, population_at_step * self.population_growth_brake, 
                    self.population_growth_brake, self.initial_population, self.ratio, idx + 1)
                message = 'value({}) < percentile({}) at epoch({}) with population({})'.format(
                    trial_value, percentile, decisive_epoch, population_message)
                self._resume(trial_id, message)

    def _resume(self, trial_id, message):
        task = self.mahler_client._create_shallow_task(trial_id)
        if task.status.name != 'Suspended':
            return
        try:
            print(task.id, task.status)
            print(message)
            self.mahler_client.resume(task, message)
        except (ValueError, RaceCondition):
            pass
