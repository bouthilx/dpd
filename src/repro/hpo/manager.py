from queue import Empty, Queue
from typing import Callable
import itertools
import logging
import datetime
from multiprocessing import Manager, Process

from repro.utils.checkpoint import CheckPointer
from repro.utils.bcolors import BColors


logger = logging.getLogger(__name__)

bcolors = BColors()


def fix_python_mp():
    # https://stackoverflow.com/questions/46779860/multiprocessing-managers-and-custom-classes
    # Backup original AutoProxy function
    import multiprocessing.managers as managers
    backup_autoproxy = managers.AutoProxy

    def redefined_autoproxy(token, serializer, manager=None, authkey=None, exposed=None,
                            incref=True, manager_owned=True):
        # Calling original AutoProxy without the unwanted key argument
        return backup_autoproxy(token, serializer, manager, authkey, exposed, incref)

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


def _get_objective(trial):
    last_results = trial.get_last_results()
    if last_results:
        return last_results[-1]["objective"]

    return None


class HPOManager:
    """ Manage a series of task - trials, it can create, suspend and resume those trials
        in function of the results it is receiving/observing through time
     """
    def __init__(self, resource_manager, dispatcher, task: Callable, trial_factory: Callable,
                 max_trials: int, workers: int, checkpoints=True):
        """
        :param task: Task that uses the HPO parameters and return results
        """
        self.manager = Manager()
        self.task = task
        self.trial_factory = trial_factory
        self.max_trials = max_trials
        # TODO: At the end of HPO, we may need less workers. This should be scaled for
        #       self.resource_manager.
        self.workers = workers

        self.resource_manager = resource_manager
        self.dispatcher = dispatcher

        self.running_trials = set()
        self.suspended_trials = set()
        self.to_be_suspended_trials = set()
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

        # List of ordered trials by creation time
        # This holds the object in themselves
        self.trials = []
        self.request_timestamps = {}
        # Used to do quick lookups in self.trials
        self.trial_lookup = {}

    def _insert_trial(self, trial):
        self.trials.append(trial)
        self.trial_lookup[trial.id] = trial

    def insert_component(self, obj):
        self.components.append(obj)

    def enable_checkpoints(self, name=None, **kwargs):
        chk = CheckPointer(**kwargs)
        chk.checkpoint_this(self, name=name)
        self.insert_component(chk)

    def run(self):
        self.resource_manager.start()
        self.param_service.start()

        try:
            while True:
                # receive the latest results from the trials & check if some trials need to be
                # suspended
                any_change = False
                stop_count = self.receive_and_suspend()
                any_change = any_change or stop_count > 0

                if self.dispatcher.is_completed() or self.trial_count >= self.max_trials:
                    logging.debug('HPO completed. Breaking out of run loop.')
                    break

                # if so check if one needs to be resumed
                resumed_count = self.resume_pending(self.workers - len(self.running_trials))
                any_change = any_change or resumed_count > 0

                # if a trial was suspended and none was resumed spawn a new trial
                started_count = self.start_new_trials()
                any_change = any_change or started_count > 0

                # Create new trial if we don't have enough
                missing = self.workers - len(self.running_trials)
                missing -= self.pending_params
                missing = min(self.max_trials - self.trial_count, missing)

                if missing > 0 and self.pending_params < 1:
                    self._queue_suggest()
                    self.pending_params = 1
                    self.trial_count += 1

                for comp in self.components:
                    comp.run()

                if any_change:
                    logger.debug(f'{bcolors.OKGREEN}running{bcolors.ENDC}:{len(self.running_trials)} '
                                 f'{bcolors.WARNING}suspended{bcolors.ENDC}:{len(self.suspended_trials)} '
                                 f'{bcolors.OKBLUE}completed{bcolors.ENDC}:{len(self.finished_trials)}')

            logger.debug('Waiting for running trials to complete.')
            # TODO: Shouldn't we suspend them?
            self._shutdown()
        except Exception as e:
            self._shutdown(False)
            raise e

        logger.debug('HPO completed')

    def get_trial(self, trial_id):
        return self.trial_lookup[trial_id]

    def _shutdown(self, gracefully=True) -> None:
        # Close self.param_service
        self.param_service.terminate()

        if gracefully:
            while self.running_trials:
                if self.receive_and_suspend():
                    logger.debug(f'{bcolors.OKGREEN}running{bcolors.ENDC}:{len(self.running_trials)} '
                                 f'{bcolors.WARNING}suspended{bcolors.ENDC}:{len(self.suspended_trials)} '
                                 f'{bcolors.OKBLUE}completed{bcolors.ENDC}:{len(self.finished_trials)}')

        for trial_id in self.running_trials:
            self.get_trial(trial_id).stop()

        self.manager.shutdown()
        self.resource_manager.terminate()

    def _queue_suggest(self) -> None:
        """ dummy function used to queue an async call to self.suggest """
        args = self.dispatcher.make_suggest_parameters()
        trial_id = args['trial_id']
        self.request_timestamps[trial_id] = str(datetime.datetime.utcnow())
        self.param_input_queue.put(args)

    def start_new_trials(self) -> int:
        """ check if some params are available in the queue, and create a trial if so.

            returns the number of started trials """

        started = 0
        while True:
            try:
                trial_id, params = self.param_output_queue.get(True, timeout=0.01)
                suggest_timestamp = str(datetime.datetime.utcnow())
                self.pending_params -= 1
                started += 1

                trial = self.trial_factory(id=trial_id, task=self.task, params=params, queue=self.manager.Queue())
                self._insert_trial(trial)

                request_time = self.request_timestamps.pop(trial_id, None)
                if request_time is None:
                    logger.error('Request timestamp is missing!')

                trial.insert_timestamp('request', request_time)
                trial.insert_timestamp('suggest', suggest_timestamp)

                trial.start()
                logger.debug(f'{trial.id} {bcolors.OKGREEN}started{bcolors.ENDC}')
                self.running_trials.add(trial_id)

            except Empty:
                return started

    def receive_and_suspend(self) -> int:
        """ iterates through running trials removing finished ones and
            checks if the running trials should be suspended

            returns the numbers of suspended trials"""

        to_be_suspended = set()
        is_finished = set()
        lost = set()

        for trial_id in self.running_trials:
            trial = self.get_trial(trial_id)
            result = trial.receive()

            if result is not None:
                for r in result:
                    trial.insert_timestamp('observe', r.get('callback_timestamp'))
                    trial.results.append(r)

                self.dispatcher.observe(trial, result)

            if trial.is_alive() and trial_id not in self.to_be_suspended_trials and self.dispatcher.should_suspend(trial_id):
                logger.debug(f'{trial.id} {bcolors.WARNING}suspending{bcolors.ENDC} {_get_objective(trial)}')
                trial.stop()
                to_be_suspended.add(trial_id)

            elif trial.has_finished():
                logger.debug(f'{trial.id} {bcolors.OKBLUE}completed{bcolors.ENDC} {_get_objective(trial)}')
                is_finished.add(trial_id)
                trial.insert_timestamp('finished')

            # Trial was lost
            elif not trial.is_alive() and trial_id not in self.to_be_suspended_trials:
                logger.debug(f'{trial.id} {bcolors.FAIL}lost{bcolors.ENDC} {_get_objective(trial)}')
                trial.insert_timestamp('lost')
                to_be_suspended.add(trial_id)

        for trial_id in is_finished:
            self.finished_trials.add(trial_id)
            # Maybe it was supposed to be suspended but completed before it could suspend.
            self.to_be_suspended_trials.discard(trial_id)
            self.running_trials.discard(trial_id)

        # NOTE: The trial will be detected as stopped later on anyway
        #       This delay is accorded so that high latency trials have a chance 
        #       to suspend properly
        for trial_id in (to_be_suspended | self.to_be_suspended_trials):
            trial = self.get_trial(trial_id)

            if not trial.is_alive():
                logger.debug(f'{trial.id} {bcolors.WARNING}suspended{bcolors.ENDC}')
                self.suspended_trials.add(trial_id)
                self.to_be_suspended_trials.discard(trial_id)
                self.running_trials.discard(trial_id)
            else:
                self.to_be_suspended_trials.add(trial_id)

        return len(to_be_suspended) + len(is_finished)

    def resume_pending(self, count) -> int:
        """ iterates through the suspended trials and check if they should be resumed

            returns the numbers of resumed trials """

        to_be_resumed = set()
        is_finished = set()

        available_workers = self.workers - len(self.running_trials) - self.pending_params

        for trial_id in self.suspended_trials:
            trial = self.get_trial(trial_id)

            if trial.has_finished():
                is_finished.add(trial_id)
                available_workers += 1

            elif available_workers > 0 and self.dispatcher.should_resume(trial_id):
                to_be_resumed.add(trial_id)
                available_workers -= 1

                if len(to_be_resumed) == count:
                    break

        for trial_id in to_be_resumed:
            trial = self.get_trial(trial_id)
            self.suspended_trials.discard(trial_id)
            trial.start()
            logger.debug(f'{trial.id} {bcolors.OKGREEN}resumed{bcolors.ENDC} {_get_objective(trial)}')
            self.running_trials.add(trial_id)

        for trial_id in is_finished:
            self.finished_trials.add(trial_id)
            self.suspended_trials.discard(trial_id)

        return len(to_be_resumed)
