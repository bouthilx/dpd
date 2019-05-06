from queue import Empty, Queue
from typing import Callable
import itertools
import logging

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
        self.resource_manager.start()
        self.param_service.start()

        try:
            while True:
                # receive the latest results from the trials & check if some trials need to be
                # suspended
                any_change = False
                stop_count = self.receive_and_suspend()
                any_change = any_change or stop_count > 0

                if self.dispatcher.is_completed():
                    logging.info('HPO completed. Breaking out of run loop.')
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
                    logger.info(f'{bcolors.OKGREEN}running{bcolors.ENDC}:{len(self.running_trials)} '
                                f'{bcolors.WARNING}suspended{bcolors.ENDC}:{len(self.suspended_trials)} '
                                f'{bcolors.OKBLUE}completed{bcolors.ENDC}:{len(self.finished_trials)}')

            logger.info('Waiting for running trials to complete.')
            # TODO: Shouldn't we suspend them?
            self._shutdown()
        except Exception as e:
            self._shutdown(False)
            raise e

        logger.info('HPO completed')

    def _shutdown(self, gracefully=True) -> None:
        # Close self.param_service
        self.param_service.terminate()

        if gracefully:
            while self.running_trials:
                if self.receive_and_suspend():
                    logger.info(f'{bcolors.OKGREEN}running{bcolors.ENDC}:{len(self.running_trials)} '
                                f'{bcolors.WARNING}suspended{bcolors.ENDC}:{len(self.suspended_trials)} '
                                f'{bcolors.OKBLUE}completed{bcolors.ENDC}:{len(self.finished_trials)}')

        for trial in self.running_trials:
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

                trial = self.trial_factory(id=trial_id, task=self.task, params=params,
                                           queue=self.manager.Queue())
                trial.start()
                logger.info(f'{trial.id} {bcolors.OKGREEN}started{bcolors.ENDC}')
                self.running_trials.add(trial)

            except Empty:
                return started

    def receive_and_suspend(self) -> int:
        """ iterates through running trials removing finished ones and
            checks if the running trials should be suspended

            returns the numbers of suspended trials"""

        to_be_suspended = set()
        is_finished = set()
        lost = set()

        for trial in self.running_trials:
            result = trial.receive()

            if result is not None:
                self.dispatcher.observe(trial, result)

            if trial.is_alive() and trial not in self.to_be_suspended_trials and self.dispatcher.should_suspend(trial.id):
                logger.info(f'{trial.id} {bcolors.WARNING}suspending{bcolors.ENDC} {trial.get_last_results()[-1]["objective"]}')
                trial.stop()
                to_be_suspended.add(trial)

            elif trial.has_finished():
                logger.info(f'{trial.id} {bcolors.OKBLUE}completed{bcolors.ENDC}{trial.get_last_results()[-1]["objective"]}')
                is_finished.add(trial)

            # Trial was lost
            elif not trial.is_alive() and trial not in self.to_be_suspended_trials:
                logger.info(f'{trial.id} {bcolors.FAIL}lost{bcolors.ENDC} {trial.get_last_results()[-1]["objective"]}')
                to_be_suspended.add(trial)

        for trial in is_finished:
            self.finished_trials.add(trial)
            # Maybe it was supposed to be suspended but completed before it could suspend.
            self.to_be_suspended_trials.discard(trial)
            self.running_trials.discard(trial)

        # NOTE: The trial will be detected as stopped later on anyway
        #       This delay is accorded so that high latency trials have a chance 
        #       to suspend properly
        for trial in (to_be_suspended | self.to_be_suspended_trials):
            if not trial.is_alive():
                logger.info(f'{trial.id} {bcolors.WARNING}suspended{bcolors.ENDC}')
                self.suspended_trials.add(trial)
                self.to_be_suspended_trials.discard(trial)
                self.running_trials.discard(trial)
            else:
                self.to_be_suspended_trials.add(trial)

        return len(to_be_suspended) + len(is_finished)

    def resume_pending(self, count) -> int:
        """ iterates through the suspended trials and check if they should be resumed

            returns the numbers of resumed trials """

        to_be_resumed = set()
        is_finished = set()

        for trial in self.suspended_trials:
            if trial.has_finished():
                is_finished.add(trial)
                
            elif self.dispatcher.should_resume(trial.id):
                to_be_resumed.add(trial)

                if len(to_be_resumed) == count:
                    break

        for trial in to_be_resumed:
            self.suspended_trials.discard(trial)
            trial.start()
            logger.info(f'{trial.id} {bcolors.OKGREEN}resumed{bcolors.ENDC} {trial.get_last_results()[-1]["objective"]}')
            self.running_trials.add(trial)

        for trial in is_finished:
            self.finished_trials.add(trial)
            self.suspended_trials.discard(trial)

        return len(to_be_resumed)
