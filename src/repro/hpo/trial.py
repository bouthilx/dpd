import bisect
import copy
import datetime
import functools
from queue import Empty as EmptyQueueException
from multiprocessing import Queue, Process
from typing import Optional, Callable, Dict, Tuple


def callback(queue, **kwargs):
    kwargs['callback_timestamp'] = str(datetime.datetime.utcnow())
    queue.put(kwargs)


class Trial:
    """ A trial represent an experience in progress, it holds all the necessary information to stop it if it is
        in progress or resume it if it was suspended """

    def __init__(self, id: str, task: Callable[[Dict[str, any]], None], params: Dict[str, any], queue: Queue):
        self.id = id
        self.task = task
        self.params = copy.deepcopy(params)
        self.kwargs = params
        self.queue: Queue = queue
        self.kwargs['callback'] = functools.partial(callback, queue=queue)
        self.process: Optional[Process] = None
        self.latest_results = None
        self._timestamps = []
        self.results = []
        self.insert_timestamp('creation')

    @property
    def timestamps(self):
        return self._timestamps

    def is_alive(self) -> bool:
        return self.process and self.process.is_alive()

    def has_finished(self) -> bool:
        return self.latest_results and self.latest_results[-1].get('finished')

    def is_suspended(self) -> bool:
        return self.process is None and not self.has_finished()

    def start(self, device) -> None:
        """ start or resume a process if it is not already running"""
        if not self.is_alive():
            self.kwargs['device'] = device
            self.process = Process(target=self.task, kwargs=self.kwargs)
            self.process.start()
            self.insert_timestamp('start')

    def stop(self, safe=False) -> None:
        """ stop the trial in progress if safe is true it will wait until the process exit """
        if self.process:
            self.process.terminate()
            self.insert_timestamp('stop')
            if safe:
                self.process.join()
            self.process = None

    def get_last_results(self) -> Tuple[Dict[str, any], ...]:
        """ return the last non null result that was received """
        return self.latest_results

    def receive(self) -> Optional[Tuple[Dict[str, any], ...]]:
        """ check if results are ready, if not return None.
            This function can return multiple results if it has not been called in a long time """
        obs = []
        while True:
            try:
                obs.append(self.queue.get(True, timeout=0.01))
            except EmptyQueueException:
                if len(obs) == 0:
                    return None

                self.latest_results = tuple(obs)
                return tuple(obs)

    def insert_timestamp(self, name, time=None):
        if time is None:
            time = str(datetime.datetime.utcnow())
        idx = bisect.bisect_left([t[1] for t in self._timestamps], time)
        self._timestamps.insert(idx, (name, time))

    def to_dict(self):
        return {
            'results': self.results,
            'timestamps': self.timestamps,
            'params': self.params,
            'id': self.id
        }


def build(id: str, task: Callable[[Dict[str, any]], None], params: Dict[str, any], queue: Queue,
          **kwargs):
    return Trial(id, task, params, queue)
