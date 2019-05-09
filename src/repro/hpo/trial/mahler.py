from multiprocessing import Process, Manager
from queue import Queue, Empty
from typing import Callable, Dict, List, Optional
import bisect
import copy
import datetime
import functools
import logging
import pickle
import signal
import time

try:
    import mahler.client as mahler
    from mahler.core.utils.errors import RaceCondition
except ImportError:
    mahler = None
    RaceCondition = Exception

from repro.hpo.trial.builtin import Trial
from repro.utils.bcolors import BColors


logger = logging.getLogger(__name__)

bcolors = BColors()


# This will be run remotely with Mahler
def mahler_callback(mahler_config, **kwargs):
    kwargs['callback_timestamp'] = str(datetime.datetime.utcnow())

    logger = logging.getLogger(__name__ + '.callback')
    mahler_client = mahler.Client(**mahler_config)
    task_id = mahler.get_current_task_id()
    mahler_client.add_metric(task_id, kwargs, metric_type='hpo', force=True)
    logger.debug(f'{task_id} add metric {kwargs}')
    mahler_client.close()

    # TODO: For mahler,
    # Start will register a task in mahler and start a monitoring process
    # The monitoring procell will listen to task.metrics[hpo] a report anything new to the queue.
    # Map task status from mahler to Trial interface.


def append_signal(sig, f):

    old = None
    if callable(signal.getsignal(sig)):
        old = signal.getsignal(sig)

    def helper(*args, **kwargs):
        if old is not None:
            old(*args, **kwargs)
        f(*args, **kwargs)

    signal.signal(sig, helper)


# TODO: Use mahler.scheduler.remoteflow to submit N workers and make sure there is always N workers
#       available (that will be in the HPOManager)
class MahlerProcess(Process):
    def __init__(self, trial_id, task, queue, arguments, operator_kwargs):
        super(Process, self).__init__()
        self.queue = queue
        self.trial_id = trial_id
        self.operator = task
        self.arguments = arguments
        # Container, tags, resource, etc.
        self.operator_kwargs = copy.deepcopy(operator_kwargs)
        self.operator_kwargs.setdefault('tags', [])
        self.operator_kwargs['tags'] = operator_kwargs['tags'] + [trial_id] + ['worker']
        self.task = None
        self.mahler_client = None
        append_signal(signal.SIGTERM, self.suspend)

    def init(self):
        self.metrics = []
        self.suspended = False
        self.mahler_client = mahler.Client()
        self.fetch_or_register()

    def register(self):
        callback = self.arguments.pop('callback')
        task = self.operator.delay(callback=pickle.dumps(callback), **self.arguments)
        self.mahler_client.register(task, **self.operator_kwargs)
        return task

    def fetch_or_register(self):
        # TODO: Turn this into name. This will need addition of name in query API and indexes in db
        tasks = list(self.mahler_client.find(tags=[self.trial_id]))
        if len(tasks) > 1:
            logger.error(f'{self.trial_id}:{self.pid} {len(tasks)} tasks assigned to this trial id')
            raise RuntimeError(f'Two tasks are assigned the trial id {self.trial_id}')
        elif tasks:
            logger.info(f'{self.trial_id}:{self.pid} Fetching from mahler')
            self.task = tasks[0]
            self.resume()
        else:
            logger.info('{} Registering to mahler {}'.format(
                self.trial_id, ';'.join(self.operator_kwargs['tags'])))
            self.task = self.register()

    def transmit_new_metrics(self):
        task_metrics = self.task.get_recent_metrics()['hpo']
        count = 0
        while len(task_metrics) > len(self.metrics):
            new_metrics = task_metrics[len(self.metrics)]
            self.metrics.append(new_metrics)
            self.queue.put(new_metrics)
            logger.debug(f'{self.trial_id}:{self.pid} New metric {new_metrics["step"]}')
            count += 1

        return count

    def queue_metrics(self):
        while True:
            self.transmit_new_metrics()
            time.sleep(1)

            remaining = self.queue.qsize()
            if not remaining:
                break

            logger.info(f'{self.trial_id}:{self.pid} {remaining} remaining in queue')

    def run(self):
        self.init()

        # TODO: Detect in HPOManager if all tasks are crashing. If so, stop execution of manager.

        status = self.task.get_recent_status()
        # NOTE: We may have lost metric reports between suspend() and the actual suspension of the
        #       remote process. Therefore, we report again all metrics here and delegate to the
        #       dispatcher the responlability of merging allready reported metrics instead of
        #       duplicating them.
        last_status = None
        while status.name not in ['Suspended', 'Cancelled', 'Completed']:
            logger.debug(f'{self.trial_id}:{self.pid} {status}')
            if last_status is None or last_status.name != status.name:
                logger.info(f'{self.trial_id}:{self.pid} {status}')

            if status.name == 'Running':
                self.queue_metrics()
            # if self.suspended:
            #     break
            # NOTE: We assume mahler is only used for long run tasks, hence why waiting at least 5
            #       seconds make sense
            time.sleep(5)
            last_status = status
            status = self.task.get_recent_status()

        self.queue_metrics()

        logger.info(f'{self.trial_id}:{self.pid} stopping with status {status}')

    def suspend(self, signum=None, frame=None, level=0, status=None):
        if level > 10:
            logger.error(f'{self.trial_id}:{self.pid} {bcolors.WARNING}cannot suspend{bcolors.ENDC}')

        if self.task:
            logger.info(f'{self.trial_id}:{self.pid} Signaling suspend to mahler ({level}:{status})')
            try:
                self.mahler_client.suspend(self.task if level == 0 else self.task.id,
                                           'Suspended by dispatcher')
                # self.suspended = True
            except (ValueError, RaceCondition) as e:
                logger.info(f'{self.trial_id}:{self.pid} {e}')
                status = self.task.get_recent_status()
                if status.name in ['OnHold', 'Queued', 'Running', 'Reserved']:
                    self.suspend(signum, frame, level=level + 1, status=status)

    def resume(self, level=0, status=None):
        if level > 10:
            logger.error(f'{self.trial_id}:{self.pid} {bcolors.WARNING}cannot resume{bcolors.ENDC}')

        if self.task:
            logger.info(f'{self.trial_id}:{self.pid} Signaling resume to mahler ({level}:{status})')
            try:
                self.mahler_client.resume(self.task if level == 0 else self.task.id,
                                          'Resumed by dispatcher')
            except (ValueError, RaceCondition):
                status = self.task.get_recent_status()
                if status.name not in ['OnHold', 'Queued', 'Running', 'Reserved', 'Completed']:
                    self.resume(level=level + 1, status=status)


class MahlerTrial(Trial):

    def __init__(self, id: str, task: Callable[[Dict[str, any]], None], params: Dict[str, any],
                 queue: Queue, tags: List[str], container: Optional[str] = None):
        super(MahlerTrial, self).__init__(id, task, params, queue)
        self.kwargs['callback'] = functools.partial(mahler_callback, mahler_config={})
        self.operator_kwargs = dict(tags=tags, container=container)
        self.process = None
        # self.manager = Manager()
        # self.event_queue = self.manager.Queue()

    @property
    def timestamps(self):
        mahler_client = mahler.Client()
        tasks = list(mahler_client.find(tags=[self.id]))
        if len(tasks) > 1:
            logger.error(f'{self.trial_id}:{self.pid} {len(tasks)} tasks assigned to this trial id')
            raise RuntimeError(f'Two tasks are assigned the trial id {self.trial_id}')
        elif not tasks:
            return self._timestamps

        def is_running():
            return timestamps and timestamps[-1][0] == 'start'

        def is_stopped():
            return ((not timestamps or timestamps[-1][0] in ['creation', 'stop', 'lost']) or
                    (status[-1] != 'Running'))

        events = tasks[0]._status.events
        del tasks
        mahler_client.close()
    
        _timestamps = copy.copy(self._timestamps)
        timestamps = []
        status = []
        for event in events:
            while _timestamps and _timestamps[0][1] < str(event['runtime_timestamp']):
                timestamps.append(_timestamps.pop(0))

            if event['item']['name'] == 'Running' and is_stopped():
                if timestamps[-1][0] == 'observe':
                    timestamps.append(('stop', timestamps[-1][1]))
                timestamps.append(('start', str(event['runtime_timestamp'])))

            status.append(event['item']['name'])

        if status[-1] != 'Running':
            timestamps.append(('stop', timestamps[-1][1]))

        return timestamps

    def start(self) -> None:
        """start or resume a process if it is not already running"""
        if not self.is_alive():
            self.process = MahlerProcess(
                trial_id=self.id, task=self.task, queue=self.queue, # event_queue=self.event_queue,
                arguments=self.kwargs, operator_kwargs=self.operator_kwargs)
            self.process.start()

    # def is_alive(self):
    #     return self.mahler_task.get_recent_status().name in ['OnHold', 'Queued', 'Running']

    # def has_finished(self):
    #     return self.mahler_task.get_recent_status().name == 'Completed'

    # def is_suspended(self):
    #     return self.mahler_task.get_recent_status().name == 'Suspended'

    # def start(self):
    #     # TODO: either
    #     #       - Register in mahler
    #     #       - Resume suspended task

    def stop(self, safe=False):
        if self.is_alive():
            self.process.terminate()
            if safe:
                self.process.join()
                self.process = None

if mahler:
    def build(id: str, task: Callable[[Dict[str, any]], None], params: Dict[str, any],
              queue: Queue, tags: List[str], container: Optional[str] = None):
        return MahlerTrial(id, task, params, queue, tags, container)
