import copy
from repro.hpo.trial import Trial

from typing import Callable, Dict, List
from queue import Empty as EmptyQueueException
from multiprocessing import Queue, Manager, Process


def _slow_function_service(slow_fun, in_queue: Queue, out_queue: Queue):
    """ function to be called inside a subprocess, essentially implement async but with multiprocess
        the parent process should send arguments to the in_queue the arguments will be passed down to the function
        to be called.
    """
    call_id = 0

    while True:
        call_id += 1
        kwargs = in_queue.get(block=True)
        result = slow_fun(**kwargs)
        out_queue.put(result)


class AbstractDispatcher:
    """ Manage a series of task - trials, it can create, suspend and resume those trials
        in function of the results it is receiving/observing through time
     """

    def __init__(self, task: Callable, workers: int):
        """
        :param task: Task that uses the HPO parameters and return results
        """
        self.manager = Manager()
        self.task = task
        self.workers = workers

        self.running_trials = set()
        self.suspended_trials = set()
        self.finished_trials = set()

        self.param_input_queue: Queue = self.manager.Queue()
        self.param_output_queue: Queue = self.manager.Queue()
        self.param_service = Process(
            target=_slow_function_service,
            args=(self.suggest, self.param_input_queue, self.param_output_queue)
        )
        self.pending_params = 0

    def run(self):
        self.param_service.start()

        try:
            while True:
                # receive the latest results from the trials & check if some trials need to be suspended
                suspended_count = self.receive_and_suspend()

                if self.is_completed():
                    break

                # if so check if one needs to be resumed
                resumed_count = self.resume_pending(suspended_count)

                # if a trial was suspended and none was resumed spawn a new trial
                started_count = self.start_new_trials()

                # Create new trial if we don't have enough
                missing = self.workers - (suspended_count - resumed_count - started_count)
                missing -= self.pending_params

                if missing > 0:
                    for i in range(0, missing):
                        self._suggest()
                        self.pending_params += 1

            self._shutdown()
        except Exception as e:
            self._shutdown(False)
            raise e

    def _shutdown(self, gracefully=True) -> None:
        # Close self.param_service
        self.param_service.terminate()

        for trial in self.running_trials:
            if gracefully:
                trial.process.join()
            else:
                trial.stop()

        self.manager.shutdown()

    def _suggest(self) -> None:
        """ dummy function used to queue an async call to self.suggest """
        self.param_input_queue.put(self.make_suggest_parameters())

    def start_new_trials(self) -> int:
        """ check if some params are available in the queue, and create a trial if so.

            returns the number of started trials """

        started = 0
        while True:
            try:
                params = self.param_output_queue.get(True, timeout=0.01)
                self.pending_params -= 1
                started += 1

                trial = Trial(self.task, params, self.manager.Queue())
                trial.start()
                self.running_trials.add(trial)

            except EmptyQueueException:
                return started

    def receive_and_suspend(self) -> int:
        """ iterates through running trials removing finished ones and
            checks if the running trials should be suspended

            returns the numbers of suspended trials"""

        to_be_suspended = set()
        is_finished = set()

        for trial in self.running_trials:
            result = trial.receive()

            if result is not None and trial.is_alive():
                self.observe(trial, result)

                if self.should_suspend(trial):
                    trial.stop()
                    to_be_suspended.add(trial)

            elif trial.has_finished():
                is_finished.add(trial)

            elif not trial.is_alive():
                to_be_suspended.add(trial)

        for trial in to_be_suspended:
            self.suspended_trials.add(trial)
            self.running_trials.discard(trial)

        for trial in is_finished:
            self.observe(trial, trial.get_last_results())
            self.finished_trials.add(trial)
            self.running_trials.discard(trial)

        return len(to_be_suspended)

    def resume_pending(self, count) -> int:
        """ iterates through the suspended trials and check if they should be resumed

            returns the numbers of resumed trials """

        to_be_resumed = set()

        for trial in self.suspended_trials:
            if self.should_resume(trial):
                to_be_resumed.add(trial)

                if len(to_be_resumed) == count:
                    break

        for trial in to_be_resumed:
            self.suspended_trials.discard(trial)
            trial.start()
            self.running_trials.add(trial)

        return len(to_be_resumed)

    def make_suggest_parameters(self) -> Dict[str, any]:
        """ return the parameters needed by `self.suggest`"""
        return {}

    def should_resume(self, trial) -> bool:
        raise NotImplementedError()

    def should_suspend(self, trial) -> bool:
        raise NotImplementedError()

    def suggest(self, **kwargs) -> Dict[str, any]:
        """return a dictionary of parameters to use for the `self.task`"""
        raise NotImplementedError()

    def observe(self, trial, results: List[Dict[str, any]]) -> None:
        raise NotImplementedError()

    def is_completed(self) -> bool:
        raise NotImplementedError()


class HPODispatcher(AbstractDispatcher):

    def __init__(self, hpo, task: Callable, workers: int):
        super(HPODispatcher, self).__init__(task, workers)
        self.hpo = hpo
        self.trial_count = 0
        self.seeds = []
        self.pending_observe = False    # We need more observation to sample different parameters
        self.last_params = None          # Last parameters that were sampled

    def should_resume(self, trial) -> bool:
        return not trial.has_finished()

    def should_suspend(self, trial) -> bool:
        return False

    def make_suggest_parameters(self) -> Dict[str, any]:
        """ return the parameters needed by `self.suggest`"""
        kwargs = dict(seed=self.seeds[self.trial_count])
        if self.pending_observe:
            kwargs['observation'] = [dict(params=self.last_params, objective=99999)]

        self.pending_observe = True
        self.last_params = kwargs
        return kwargs

    def suggest(self, seed, observations) -> Dict[str, any]:
        if observations is None:
            self.trial_count += 1
            self.last_params = self.hpo.get_params(seed=seed)
            return self.last_params

        configurator = copy.deepcopy(self.hpo)
        configurator.observe(observations)

        self.hpo.get_params(seed)
        self.last_params = configurator.get_params(seed=seed)
        return self.last_params

    def observe(self, trial, results) -> None:
        self.hpo.observe([dict(params=trial.params, objective=result['objective']) for result in results])
        self.pending_observe = False

    def is_completed(self) -> bool:
        return self.hpo.is_completed() or self.trial_count >= self.hpo.max_trial




