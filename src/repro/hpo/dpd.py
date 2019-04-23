import argparse
import bisect
import bson
import datetime
import os
import time

import cotyledon

import scipy.integrate

from bson import objectid

import numpy

try:
    import mahler.client as mahler
    from mahler.core.utils.errors import SignalInterruptTask, SignalSuspend, RaceCondition
except ImportError:
    mahler = None
    SignalSuspend = None


class DPDManager(cotyledon.ServiceManager):
    def __init__(self, tags, output, timestamp=None):
        super(DPDManager, self).__init__()

        if timestamp:
            timestamp = bson.ObjectId.from_datetime(timestamp)

        self.add(DPDFire, args=(tags, output, timestamp, ))
        self.add(DPDIce, args=(tags, output, timestamp, ))


class DPDRock(cotyledon.Service):
    """
    DPD daemon to compile metrics.
    """
    name = 'dpd-rock'

    def __init__(self, worker_id, tags, output, timestamp=None):
        self.id = worker_id
        self.experiments = {}
        self.trials = {}
        self.first_task_timestamp = None
        self.task_timestamp = timestamp
        self.metric_timestamp = timestamp
        self.tags = tags

        self.output = output
        self.files = {}

        self._backoff = 0

        if timestamp:
            timestamp = str(timestamp.generation_time)

        self.print('Started with {}'.format({'tags': tags, 'timestamp': timestamp}))

    def print(self, *msg, name=None):
        if self.output is None:
            self.print_to_stdout(*msg, name=name)
        else:
            self.print_to_file(*msg, name=name)

    def print_to_stdout(self, *msg, name):
        header = '{}({})'.format(self.name, self.id)
        if name:
            header += '-' + name

        print(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), header, '\t', *msg)

    def print_to_file(self, *msg, name):
        filename = '{}-{}'.format(self.name, self.id)
        if name:
            filename += '-' + name

        filename += '.log'

        filepath = os.path.join(self.output, filename)

        if filename not in self.files:
            self.files[filename] = open(filename, 'a')

        f = self.files[out]

        print(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), '\t', *msg, file=f)

    @property
    def db_client(self):
        return self.mahler_client.registrar._db._db

    def open_connection(self):
        self.mahler_client = mahler.Client()

    def add_experiment(self, tags, trial_id, rules):
        if tags not in self.experiments:
            self.print('Adding experiment {}'.format(tags))
            self.experiments[tags] = Experiment(tags=tags, **rules)

        self.trials[trial_id] = self.experiments[tags]

    def add_trial(self, tags, trial_id, report_timestamp, status, rules):
        self.add_experiment(tags, trial_id, rules)

        trial_added = self.experiments[tags].add_trial(trial_id, status)

        self.task_timestamp = report_timestamp
        trial_timestamp = bson.ObjectId(trial_id)

        if self.first_task_timestamp:
            self.first_task_timestamp = min(self.first_task_timestamp, trial_timestamp)
        else:
            self.first_task_timestamp = trial_timestamp

        if self.metric_timestamp:
            self.metric_timestamp = min(self.task_timestamp, self.metric_timestamp)

        self.print('{:>10} {} {} {} {}'.format(
            'Add' if trial_added else 'Update', trial_id,
            self.first_task_timestamp, self.task_timestamp, self.metric_timestamp), name=tags)

        return trial_added

    def add_metric(self, trial_id, metric_id, step, value):

        experiment = self.trials.get(trial_id, None)

        if experiment is None:
            return False

        metric_added = experiment.add_metric(trial_id, step, value)
        self.metric_timestamp = metric_id

        self.print('    {:>10} {} {} {} {}'.format(
            'Add' if metric_added else 'Update',
            trial_id, step, value, self.metric_timestamp), name=experiment.tags)

        return metric_added

    def update_trials(self):
        # fetch trials report based on updated timestamp
        query = {}
        if self.tags:
            query['registry.tags'] = {'$all': self.tags}

        if self.task_timestamp:
            query['registry.reported_on'] = {'$gt': objectid.ObjectId(self.task_timestamp)}

        projection = {
            '_id': 1, 'registry.reported_on': 1, 'registry.tags': 1, 'registry.status': 1,
            'arguments.stopping_rule': 1}

        trials_added = 0
        start = time.time()

        for trial in self.db_client.tasks.report.find(query, projection):
            trials_added += int(self.add_trial(
                ':'.join(sorted(trial['registry']['tags'])), str(trial['_id']),
                report_timestamp=trial['registry']['reported_on'],
                status=trial['registry']['status'],
                rules=trial['arguments']['stopping_rule']))
            # If a trial is added, maybe some metrics were loaded but discarded because trial was
            # not loaded yet. To be safe, fallback metric_timestamp
        self.print('{:10d} new trials:  {:>5f} seconds'.format(trials_added, time.time() - start))

        if self.metric_timestamp is None and self.first_task_timestamp:
            self.metric_timestamp = self.first_task_timestamp

        return trials_added

    def update_metrics(self):
        # query = {'item.type': 'stop'}
        query = {'item.type': 'stat'}

        if self.metric_timestamp:
            query['_id'] = {'$gt': self.metric_timestamp}

        projection = {'task_id': 1, 'item.value.epoch': 1, 'item.value.valid.error_rate': 1}

        metrics_added = 0
        start = time.time()

        # fetch metrics after timestamp, ignore metric if not in trial ids
        for metric in self.db_client.tasks.metrics.find(query, projection):
            stats = metric['item']['value']
            metrics_added += int(self.add_metric(
                str(metric['task_id']), metric['_id'], stats['epoch'],
                stats['valid']['error_rate']))

        self.print('{:10d} new metrics: {:>5f} seconds'.format(metrics_added, time.time() - start))

        return metrics_added

    def backoff(self, changes):
        if changes == 0:
            sleep_time = min(2 ** self._backoff, 8)
            self.print('No changes. Sleeping for {}sec'.format(sleep_time))
            time.sleep(sleep_time)
            self._backoff += 1
        else:
            self._backoff = 0


class DPDFire(DPDRock):
    """
    DPD daemon to thaw suspended trials.
    """
    name = 'dpd-fire'

    @property
    def db_client(self):
        return self.mahler_client.registrar._db._db

    def thaw(self, trial_id, message, tags):
        self.print('Attempt thaw: {}'.format(message), name=tags)
        task = self.mahler_client._create_shallow_task(trial_id)
        if task.status.name != 'Suspended':
            self.print('Not suspended, updating status: {}'.format(task.status.name), name=tags)
            self.trials[trial_id]._trial_status[trial_id] = task.status.name
            return False
        try:
            self.print('Attempt re-queuing', name=tags)
            self.mahler_client.resume(task, message)
            self.db_client.tasks.signal_status.find_one_and_update(
                {'_id': bson.ObjectId(trial_id)}, {'$set': {'status': RUNNING, 'msg': ''}})
            self.trials[trial_id]._trial_status[trial_id] = 'Queued'
            self.print('Success', name=tags)
            thawed = True
        except (ValueError, RaceCondition) as e:
            self.print('Race-condition, abort: {}'.format(str(e)), name=tags)
            thawed = False

        return thawed

    def thaw_if_possible(self, experiment, trial_id):
        self.print('Eval {} for thaw'.format(trial_id), name=experiment.tags)
        msg = experiment.should_thaw(trial_id)

        if msg:
            return self.thaw(trial_id, msg, experiment.tags)

        return False

    def thaw_trials(self):
        thawed = 0
        start_time = time.time()
        for experiment in self.experiments.values():
            for trial_id in experiment.get_resumable_trials():
                thawed += int(self.thaw_if_possible(experiment, trial_id))

        self.print('{:10d} trials thawed: {:>5f} seconds'.format(thawed, time.time() - start_time))

        return thawed

    def run(self):
        self.open_connection()

        while True:
            changes = self.update_trials()
            changes += self.update_metrics()
            changes += self.thaw_trials()

            self.backoff(changes)


COMPLETED = 0
RUNNING = 1
SUSPENDED = 2


class DPDIce(DPDRock):
    """
    DPD daemon to track metrics and suspend running trials.
    """
    name = 'dpd-ice'

    def suspend_if_needed(self, trial_id, step):
        experiment = self.trials[trial_id]
        self.print('Eval {} for icing at step {}'.format(trial_id, step), name=experiment.tags)
        msg = experiment.should_suspend(trial_id, step)
        status = SUSPENDED if msg else RUNNING
        self.print('Setting status to {}: {}'.format(status, msg), name=experiment.tags)
        self.db_client.tasks.signal_status.find_one_and_update(
            {'_id': bson.ObjectId(trial_id)},
            {'$set': {'status': status, 'msg': msg, 'step': step}},
            upsert=True)

    def add_metric(self, trial_id, metric_id, step, value):
        metric_added = super(DPDIce, self).add_metric(trial_id, metric_id, step, value)
        if metric_added and self.ready:
            self.suspend_if_needed(trial_id, step)

        return metric_added

    def run(self):
        self.open_connection()

        self.ready = False
        while True:
            changes = self.update_trials()
            changes += self.update_metrics()

            self.backoff(changes)

            if not self.ready:
                self.print('DPD server up-to-date and ready to spread some ice.')
                self.ready = True


class Experiment:
    def __init__(self, tags, initial_population=200, final_population=20, n_steps=30,
                 window_size=11, n_points=121, min_population=3, population_growth_brake=0.5):

        self.tags = tags

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
        self._trial_status = {}

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

    @property
    def status(self):
        n_completed = 0
        n_suspended = 0
        for status in self._trial_status.values():
            n_completed += int(status == 'Completed')
            n_suspended += int(status == 'Suspended')

        print(n_completed, n_suspended, self.initial_population, self.final_population)
        if ((n_completed < self.final_population) or
            (n_completed + n_suspended < self.initial_population)):

            return 'Running'

        return 'Completed'

    def get_resumable_trials(self):
        if self.status == 'Completed':
            return []

        return [trial_id for trial_id, status in self._trial_status.items()
                if status == 'Suspended']

    def add_trial(self, trial_id, status):
        added = trial_id not in self.trials

        if added:
            self.trials[trial_id] = len(self.trials)

        self._trial_status[trial_id] = status

        return added

    def add_metric(self, trial_id, step, value):
        metric_key = self.trials[trial_id]
        new_metric = bool(self.metrics[metric_key][step] < 0)
        self.metrics[metric_key][step] = value

        # NOTE: this may fail if some steps are missing in metrics and have default -1
        if step > 1:
            metrics = self.metrics[metric_key][1:step + 1]
            self.smooth_metrics[metric_key][step] = metrics[numpy.where(metrics > 0)].min()
        else:
            self.smooth_metrics[metric_key][step] = self.metrics[metric_key][step]

        return new_metric

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

    def should_thaw(self, trial_id):
        metric_index = self.trials[trial_id]
        trial_metrics = self.smooth_metrics[metric_index]
        # We can ignore epoch 0 because it is set to -1

        epochs = numpy.where(trial_metrics >= 0)[0]
        if epochs.shape[0] == 0:
            return

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
            return message

        percentile, population = self.fetch_results(last_epoch)

        if percentile is None:
            return

        trial_value = trial_metrics[last_epoch]
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
            return message

    def should_suspend(self, trial_id, epoch):
        # if epoch < self.grace:
        #     return

        if self.epochs_buffer is None:
            return

        if epoch < self.first_epoch + self.epochs_buffer:
            return

        # print("Should I skip?")
        # print(epoch)
        decisive_epochs = self.get_decisive_steps()
        # print(decisive_epochs)
        if epoch < decisive_epochs[0]:
            return

        idx = max(bisect.bisect(decisive_epochs, epoch) - 1, 0)
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
            return message
        elif percentile is None:
            return

        decisive_value = self.smooth_metrics[self.trials[trial_id], decisive_epoch]
        # print(decisive_value)
        if decisive_value > percentile:
            message = 'value({}) > percentile({}) at epoch({}) with population({})'.format(
                decisive_value, percentile, decisive_epoch, population)
            return message

    def fetch_results(self, epoch, include_task=True):
        mask = self.smooth_metrics[:, epoch] >= 0
        population = mask.sum()
        if population < self.min_population:
            return None, population

        indices = numpy.where(mask)
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


class DynamicPercentileDispatcher:

    def __init__(self, initial_population, final_population, n_steps=30, window_size=11,
                 n_points=200, min_population=3, population_growth_brake=0.5,
                 update_timeout=60):

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
        self.update_timeout = update_timeout

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

    @property
    def db_client(self):
        return self.mahler_client.registrar._db._db

    def close(self):
        self.mahler_client.close()

    def signal_step(self, stats):
        self.mahler_client.add_metric(self.task.id, stats, metric_type='stop', force=True)

    def verify(self, epoch):
        i = 0
        docs = []
        start_time = time.time()
        while time.time() - start_time < self.update_timeout:
            if i >= 0:
                print('sleeping', 2**i)
                time.sleep(2 ** i)

            i += 1
            docs = list(self.db_client.tasks.signal_status.find({'_id': self.task.id}))
            if docs and docs[0]['step'] == epoch:
                if docs[0]['status'] == SUSPENDED:
                    raise SignalSuspend(docs[0]['msg'])

                return

        raise SignalInterruptTask('DPD Timeout: {: 2.1f}s ({})'.format(
            time.time() - start_time, docs[0].get('step', -1) if docs else None))

    def signal_resume(self, epoch):
        # Make sure it does not exist, or that it was expected to resume
        docs = list(self.db_client.tasks.signal_status.find({'_id': self.task.id}))
        if docs and (docs[0]['status'] != RUNNING and docs[0]['step'] <= epoch):
            if docs[0]['status'] == SUSPENDED:
                message = 'Cannot resume: {}'.format(docs[0]['msg'])
                raise SignalSuspend(message)
            else:
                message = 'Trial was not signaled properly: {}'.format(docs[0]['status'])
                raise RuntimeError(message)
        elif docs and docs[0]['step'] > epoch:
            message = 'Progress partly lost. Starting over at epoch {}'.format(epoch)
            self.db_client.tasks.signal_status.find_one_and_update(
                {'_id': self.task.id}, {'$set': {'status': RUNNING, 'msg': message, 'step': epoch}})

    def signal_completion(self, epoch):
        self.db_client.tasks.signal_status.find_one_and_update(
            {'_id': self.task.id}, {'$set': {'status': COMPLETED, 'msg': '', 'step': epoch}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DPD server.')
    parser.add_argument(
        '--tags', nargs='*', type=str, default=['hpo'],
        help='Tags used to limit which trials may be loaded for scheduling. Default: hpo')

    parser.add_argument(
        '--output', type=str, default=None,
        help='Output dir where to write logs. If not specified, log is printed to stdout')

    parser.add_argument(
        '--cold', action='store_true',
        help='Start indexing from now, so that prior experiments are ignored.')

    options = parser.parse_args()
    tags = list(sorted(set(options.tags)))
    timestamp = datetime.datetime.utcnow() if options.cold else None
    DPDManager(tags=options.tags, output=options.output, timestamp=timestamp).run()
