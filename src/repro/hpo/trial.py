import copy
from queue import Empty as EmptyQueueException
from multiprocessing import Queue, Process
from typing import Optional, Callable, Dict, Tuple


class Trial:
    """ A trial represent an experience in progress, it holds all the necessary information to stop it if it is
        in progress or resume it if it was suspended """
    def __init__(self, task: Callable[[Dict[str, any]], None], params: Dict[str, any], queue: Queue):
        self.task = task
        self.params = copy.deepcopy(params)
        self.kwargs = params
        self.queue: Queue = queue
        self.kwargs['queue'] = queue
        self.process: Optional[Process] = None
        self.latest_results = None

    def is_alive(self) -> bool:
        return self.process and self.process.is_alive()

    def has_finished(self) -> bool:
        return self.latest_results and self.latest_results[-1].get('finished')

    def is_suspended(self) -> bool:
        return self.process is None and not self.has_finished()

    def start(self) -> None:
        """ start or resume a process if it is not already running"""
        if not self.is_alive():
            self.process = Process(target=self.task, kwargs=self.kwargs)
            self.process.start()

    def stop(self, safe=False) -> None:
        """ stop the trial in progress if safe is true it will wait until the process exit """
        if self.process:
            self.process.terminate()
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

