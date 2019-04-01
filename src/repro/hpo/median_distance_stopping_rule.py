import pprint

import numpy

try:
    import mahler.client as mahler
    from mahler.core.utils.errors import SignalSuspend
except ImportError:
    mahler = None
    SignalSuspend = None


# TODO: Add metrics to mahler tasks
#       Test with lenet


class MedianDistanceStoppingRule:

    def __init__(self, grace=5, min_population=3):
        self.grace = grace
        self.min_population = min_population

        self.trials = {}
        self.metrics = numpy.zeros((2000, 300))

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

    @property
    def db_client(self):
        return self.mahler_client.registrar._db._db

    def close(self):
        self.mahler_client.close()

    def update(self, stats):
        if self.task_id is None:
            pprint.pprint(stats)
            return

        epoch = stats['epoch']
        if epoch < self.grace:
            return

        # This will refresh the trials and metrics at the same time.
        self.thaw()
        self.mahler_client.add_metric(self.task_id, stats)

        best_value, median_distance, _ = self.fetch_results(epoch, include_task=False)
        if best_value is None:
            return

        distance = abs(best_value - stats['valid']['error_rate'])
        if distance > median_distance:
            message = 'Distance above median. (epoch:{}, dist:{}, median: {})'.format(
                epoch, distance, median_distance)
            raise SignalSuspend(message)
                
    def refresh(self):
        self._update_trials()
        self._update_metrics()
        
    def _update_trials(self):
        # fetch trials report based on updated timestamp
        query = {'tags': {'$all': self.tags}}

        if self.task_timestamp:
            query['registry.reported_on'] = {'$gt': self.task_timestamp}

        projection = {'_id': 1}

        for trial in self.db_client.tasks.report.find(query, projection):
            if trials['_id'] in self.trials:
                continue

            self.trials[trial['_id']] = len(self.trials)
            self.task_timestamp = trial['registry.reported_on']

    def _update_metrics(self):
        query = {}

        if self.metric_timestamp:
            query['_id'] = {'$gt': self.task_timestamp}

        projection = {'task_id': 1, 'item.epoch': 1, 'item.valid.error_rate': 1}

        # fetch metrics after timestamp, ignore metric if not in trial ids
        for metric in self.db_client.tasks.metric.find(query, projection):
            if metric['task_id'] not in self.trials:
                continue

            stats = metric['item']
            metric_key = self.trials[metric['task_id']]
            epoch = stats['epoch']
            self.metrics[metric_key][epoch] = stats['valid']['error_rate']
            min_values = numpy.minimum.accumulate(self.metrics[metric_key][epoch:])
            self.metrics[metric_key][epoch:] = min_values
            self.metric_timestamp = metric['_id']

    def fetch_results(self, epoch, include_task=True):
        if len(self.trials) < self.min_population:
            return None, None, None

        best = self.best[epoch]
        median_distance = numpy.median(self.metrics[:, epoch] - best)

        return best, median_distance
        
    def thaw(self):
        if self.task_id is None:
            return

        self.refresh()
        medians = dict()
        for trial_id, metric_index in self.trials.items():
            trial_metrics = self.metrics[metric_index]
            # - 1 to avoid counting metric before first epoch
            last_epoch = sum(trial_metrics > 0) - 1
            if last_epoch not in medians:
                medians[last_epoch] = self.fetch_results(last_epoch)

            best_trial_value = trial_metrics[last_epoch]
            median_distance = medians[last_epoch][1]
            distance = abs(medians[last_epoch][0] - best_trial_value)
            if distance <= median_distance:
                message = 'Distance below median. (epoch:{}, dist:{}, median: {})'.format(
                    last_epoch, distance, median_distance)
                self.mahler_client.resume(trial_id, message)
