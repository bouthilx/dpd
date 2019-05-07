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


# TODO: Add metrics to mahler tasks
#       Test with lenet


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


class MedianDistanceStoppingRule:

    def __init__(self, grace=5, min_population=3, n_steps=10):
        grace = 1
        min_population = 3
        self.grace = grace
        self.min_population = min_population
        self.n_steps = n_steps

        self.trials = {}
        # self.metrics = numpy.zeros((2000, 300))
        self.metrics = numpy.ones((3000, 200)) * -1
        self.best = numpy.zeros(61)

        self.task_timestamp = None
        self.metric_timestamp = None

        self.task_id = None
        if mahler is not None:
            self.task_id = mahler.get_current_task_id()
            if self.task_id is not None:
                self.mahler_client = mahler.Client()
                task = list(self.mahler_client.find(id=self.task_id))[0]
                self.tags = task.tags

        if self.task_id is None:
            print('WARNING: Cannot use MedianDistanceStoppingRule without mahler.')

    def get_decisive_steps(self):
        """Compute at which steps suspension should occur based on metric history"""
        # Compute variance of performances at each epoch across all trials so far.
        mask = (self.metrics >= 0)
        n_points = mask.sum(axis=0)
        # # Ignore epochs where there is less trials than min_population
        mask *= (n_points > self.min_population)
        # # Set to n_points to on 1 the epochs where there is less trials than min_population to
        # # avoid divbyzero errors
        # n_points = numpy.maximum(mask.sum(axis=0), 1)
        # mean = (self.metrics.sum(axis=0) * mask) / n_points
        # diff = (self.metrics - mean) * mask
        # var = (diff * diff).sum(axis=0) / n_points

        # print(var)
        diff = (numpy.max(self.metrics * mask, axis=0) - numpy.min(self.metrics * mask, axis=0))
        var = diff

        # print(var)

        # Compute the cummulative integral of variance over epochs
        # and pick a list of epochs so that all intervals contain the same
        # total amount of variance.
        vcum = scipy.integrate.cumtrapz(var, initial=0)
        # print(vcum)
        block = vcum[-1] / self.n_steps
        steps = []
        j = 1
        for i, accum in enumerate(vcum):
            if accum > block * j:
                j += 1
                steps.append(i)

        if not steps:
            steps.append(1)

        while len(steps) < self.n_steps:
            steps.append(steps[-1] + 1)

        print(steps)
        return steps

    @property
    def db_client(self):
        return self.mahler_client.registrar._db._db

    def close(self):
        self.mahler_client.close()

    def update(self, stats):
        if self.task_id is None:
            pprint.pprint(stats)
            return

        self.mahler_client.add_metric(self.task_id, stats)

        epoch = stats['epoch']
        # if epoch < self.grace:
        #     return
        print("Should I skip?")
        print(epoch)
        decisive_epochs = self.get_decisive_steps()
        print(decisive_epochs)
        if epoch < decisive_epochs[0]:
            return

        # This will refresh the trials and metrics at the same time.
        self.thaw(subset=10)

        decisive_epochs = self.get_decisive_steps()

        if epoch < decisive_epochs[0]:
            return

        idx = bisect.bisect_left(decisive_epochs, epoch)
        decisive_epochs = decisive_epochs[:min(idx, len(decisive_epochs) - 1)]

        for decisive_epoch in decisive_epochs:
            best_value, median_distance = self.fetch_results(decisive_epoch)

            if best_value is None:
                return

            decisive_value = self.metrics[self.trials[self.task_id], decisive_epoch]
            distance = abs(best_value - decisive_value)

            if distance > median_distance:
                message = 'Distance > median. (epoch:{}, dist:{}, median: {})'.format(
                    epoch, distance, median_distance)
                raise SignalSuspend(message)
                # self.mahler_client.suspend(self.task_id, message)

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
            min_values = numpy.minimum.accumulate(self.metrics[metric_key][epoch:])
            self.metrics[metric_key][epoch:] = min_values
            self.metric_timestamp = metric['_id']

        print('metric time', time.time() - start)

        best_index = numpy.argmin((self.metrics + 10000 * (self.metrics < 0)).min(axis=1))
        self.best = self.metrics[best_index, :]
        # print(self.trials)
        # print(self.metrics)
        # print(self.best)

    def fetch_results(self, epoch, include_task=True):
        if (self.metrics[:, epoch] >= 0).sum() < self.min_population:
            return None, None

        best = self.best[epoch]
        indices = numpy.where(self.metrics[:, epoch] >= 0)
        median_distance = numpy.median(numpy.abs(self.metrics[indices, epoch] - best))

        return best, median_distance

    def thaw(self, subset=None):
        if self.task_id is None:
            return

        self.refresh()
        decisive_epochs = self.get_decisive_steps()
        medians = dict()
        if subset:
            trial_ids = numpy.random.choice(list(self.trials.keys()), size=subset, replace=False)
        else:
            trial_ids = self.trials.keys()
    
        # TODO: Compute all based on matrix directly and back-track to trial ids.
        for trial_id in trial_ids:
            metric_index = self.trials[trial_id]
            if trial_id == self.task_id:
                continue

            trial_metrics = self.metrics[metric_index]
            # - 1 to avoid counting metric before first epoch
            last_epoch = sum(trial_metrics >= 0) - 1

            if last_epoch < 1:
                self._resume(trial_id, 'Should not be suspended before epoch 1')
                continue

            idx = bisect.bisect_left(decisive_epochs, last_epoch)
            decisive_epoch = decisive_epochs[min(idx, len(decisive_epochs) - 1)]

            if last_epoch < decisive_epoch:
                message = 'Should not be suspended before epoch {}. (Was {})'.format(
                    decisive_epoch, last_epoch)
                self._resume(trial_id, message)
                continue

            if decisive_epoch not in medians:
                medians[decisive_epoch] = self.fetch_results(decisive_epoch)

            if medians[decisive_epoch][0] is None:
                continue

            trial_value = trial_metrics[decisive_epoch]
            median_distance = medians[decisive_epoch][1]
            distance = abs(medians[decisive_epoch][0] - trial_value)
            # Deliberately avoid = median to limit unstability
            # (if one trial moves around median it would get suspended and resumed very often)
            if distance < median_distance:
                message = 'Distance < median. (epoch:{}, dist:{}, median: {})'.format(
                    decisive_epoch, distance, median_distance)
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
