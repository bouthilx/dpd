import copy
from collections import defaultdict
import itertools
import logging
import uuid
import numpy
import itertools

from collections import defaultdict
from typing import Callable, Dict, Tuple
from queue import Empty as EmptyQueueException

from repro.hpo.dispatcher.trial import Trial
from repro.hpo.configurator.base import build_configurator
from repro.utils.checkpoint import CheckPointer

from multiprocessing import Queue, Manager, Process


logger = logging.getLogger(__name__)


def fix_python_mp():
    # https://stackoverflow.com/questions/46779860/multiprocessing-managers-and-custom-classes
    # Backup original AutoProxy function
    import multiprocessing.managers as managers
    backup_autoproxy = managers.AutoProxy

    def redefined_autoproxy(token, serializer, manager=None, authkey=None, exposed=None, incref=True, manager_owned=True):
        # Calling original AutoProxy without the unwanted key argument
        return backup_autoproxy(token, serializer, manager, authkey,  exposed, incref)

    managers.AutoProxy = redefined_autoproxy


fix_python_mp()


def _slow_function_service(slow_fun, in_queue: Queue, out_queue: Queue):
    """ function to be called inside a subprocess, essentially implement async but with multiprocess
        the parent process should send arguments to the in_queue the arguments will be passed down
        to the function to be called.
    """
    call_id = 0

    while True:
        call_id += 1
        kwargs = in_queue.get(block=True)
        result = slow_fun(**kwargs)
        out_queue.put(result)


class HPOManager:
    """ Manage a series of task - trials, it can create, suspend and resume those trials
        in function of the results it is receiving/observing through time
     """
    def __init__(self, dispatcher, task: Callable, max_trials: int, workers: int, checkpoints=True):
        """
        :param task: Task that uses the HPO parameters and return results
        """
        self.manager = Manager()
        self.task = task
        self.max_trials = max_trials
        self.workers = workers

        self.dispatcher = dispatcher

        self.running_trials = set()
        self.suspended_trials = set()
        self.finished_trials = set()

        self.param_input_queue: Queue = self.manager.Queue()
        self.param_output_queue: Queue = self.manager.Queue()
        self.param_service = Process(
            target=_slow_function_service,
            args=(dispatcher.receive_and_suggest, self.param_input_queue, self.param_output_queue)
        )
        self.pending_params = 0
        self.trial_count = 0
        # Additional class we should run
        self.components = []
        if checkpoints:
            self.enable_checkpoints()

    def insert_component(self, obj):
        self.components.append(obj)

    def enable_checkpoints(self, name=None, **kwargs):
        chk = CheckPointer(**kwargs)
        chk.checkpoint_this(self, name=name)
        self.insert_component(chk)

    @property
    def trials(self):
        return itertools.chain(self.suspended_trials, self.running_trials, self.finished_trials)

    def run(self):
        self.param_service.start()

        try:
            while True:
                # receive the latest results from the trials & check if some trials need to be
                # suspended
                suspended_count = self.receive_and_suspend()

                if self.dispatcher.is_completed():
                    logging.info('HPO completed. Breaking out of run loop.')
                    break

                # if so check if one needs to be resumed
                resumed_count = self.resume_pending(suspended_count)

                # if a trial was suspended and none was resumed spawn a new trial
                started_count = self.start_new_trials()

                # Create new trial if we don't have enough
                missing = self.workers - len(self.running_trials)  # (suspended_count - resumed_count - started_count)
                missing -= self.pending_params
                missing = min(self.max_trials - self.trial_count, missing)

                if missing > 0 and self.pending_params < 1:
                    self._queue_suggest()
                    self.pending_params = 1
                    self.trial_count += 1

                for comp in self.components:
                    comp.run()

            self._shutdown()
        except Exception as e:
            self._shutdown(False)
            raise e

        logger.info('HPO completed')

    def _shutdown(self, gracefully=True) -> None:
        # Close self.param_service
        self.param_service.terminate()

        for trial in self.running_trials:
            if gracefully:
                trial.process.join()
            else:
                trial.stop()

        self.manager.shutdown()

    def _queue_suggest(self) -> None:
        """ dummy function used to queue an async call to self.suggest """
        self.param_input_queue.put(self.dispatcher.make_suggest_parameters())

    def start_new_trials(self) -> int:
        """ check if some params are available in the queue, and create a trial if so.

            returns the number of started trials """

        started = 0
        while True:
            try:
                trial_id, params = self.param_output_queue.get(True, timeout=0.01)
                self.pending_params -= 1
                started += 1

                trial = Trial(trial_id, self.task, params, self.manager.Queue())
                trial.start()
                logger.info(f'{trial.id} started')
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

            if result is not None:
                self.dispatcher.observe(trial, result)

            if trial.is_alive() and self.dispatcher.should_suspend(trial.id):
                trial.stop()
                logger.info(f'{trial.id} suspended {trial.get_last_results()[-1]["objective"]}')
                to_be_suspended.add(trial)

            elif trial.has_finished():
                logger.info(f'{trial.id} completed {trial.get_last_results()[-1]["objective"]}')
                is_finished.add(trial)

            # Trial was lost
            elif not trial.is_alive():
                logger.info(f'{trial.id} lost {trial.get_last_results()[-1]["objective"]}')
                to_be_suspended.add(trial)

        for trial in to_be_suspended:
            self.suspended_trials.add(trial)
            self.running_trials.discard(trial)

        for trial in is_finished:
            self.finished_trials.add(trial)
            self.running_trials.discard(trial)

        return len(to_be_suspended)

    def resume_pending(self, count) -> int:
        """ iterates through the suspended trials and check if they should be resumed

            returns the numbers of resumed trials """

        to_be_resumed = set()

        for trial in self.suspended_trials:
            if self.dispatcher.should_resume(trial.id):
                to_be_resumed.add(trial)

                if len(to_be_resumed) == count:
                    break

        for trial in to_be_resumed:
            self.suspended_trials.discard(trial)
            trial.start()
            logger.info(f'{trial.id} resumed {trial.get_last_results()[-1]["objective"]}')
            self.running_trials.add(trial)

        return len(to_be_resumed)


class HPODispatcher:
    def __init__(self, space: 'Space', configurator_config: Dict[str, any], max_trials: int, seed=1):
        self.space = space
        self.configurator_config = configurator_config
        self.trial_count = 0
        self.seeds = numpy.random.RandomState(seed).randint(0, 100000, size=(max_trials, ))
        self.observations: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.params = dict()
        self.buffered_observations = []
        self.finished = set()
        self.max_trials = max_trials

    def should_resume(self, trial_id) -> bool:
        raise NotImplementedError()

    def should_suspend(self, trial_id) -> bool:
        raise NotImplementedError()

    def make_suggest_parameters(self) -> Dict[str, any]:
        """ return the parameters needed by `self.suggest`"""
        kwargs = dict(
            trial_id=uuid.uuid4().hex,
            seed=self.seeds[self.trial_count],
            buffered_observations=self.buffered_observations)

        # Pre-registering bad result. This is necessary so that batch of async calls inform algos of
        # previous call that are not completed yet. We don't know
        worst_objectives = -float('inf')
        empty = True
        for objectives in self.observations.values():
            last_step = max(objectives.keys())
            if last_step >= 0:
                worst_objectives = max(objectives[last_step], worst_objectives)
                empty = False

        if empty:
            worst_objectives = 99999

        self.buffered_observations = []
        self._observe(trial_id=kwargs['trial_id'], params=None, step=-1, objective=worst_objectives)

        self.trial_count += 1

        return kwargs

    def receive_and_suggest(self, trial_id, buffered_observations, seed) -> Tuple[str, Dict[str, any]]:

        for observation in buffered_observations:
            self._observe(**observation)

        params = self.params[trial_id] = self.suggest(seed)

        return trial_id, params

    def suggest(self, seed) -> Dict[str, any]:
        self.trial_count += 1
        return self.build_configurator().get_params(seed=seed)

    def build_configurator(self):
        configurator = build_configurator(self.space, **self.configurator_config)

        for trial_id, observations in self.observations.items():
            _, objective = self.get_objective(trial_id)

            if objective is None:
                raise RuntimeError(f'Preregistration failed for trial {trial_id}')

            configurator.observe(self.params[trial_id], objective)

        return configurator

    def observe(self, trial, results) -> None:

        for result in results:
            observation = copy.deepcopy(result)
            observation.update({'trial_id': trial.id, 'params': trial.params})
            self.buffered_observations.append(observation)
            self._observe(**observation)

    def _observe(self, trial_id, params, step, objective, finished=False, **kwargs):
        if finished:
            self.finished.add(trial_id)

        self.params[trial_id] = params
        self.observations[trial_id][step] = objective

    def is_completed(self) -> bool:
        return self.hpo.is_completed() or self.trial_count >= self.hpo.max_trial

    def get_objective(self, trial_id: str) -> Tuple[int, Dict[str, any]]:
        steps = self.observations[trial_id].keys()
        if not steps:
            return None, None

        last_step = max(steps)

        return last_step, self.observations[trial_id][last_step]
