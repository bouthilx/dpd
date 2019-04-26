import copy
from queue import Empty as EmptyQueueException

from typing import Optional
from multiprocessing import Queue, Manager, Process
from repro.utils.chrono import Chrono


class Trial:
    def __init__(self, task, params, queue: Queue):
        self.task = task
        self.kwargs = params
        self.queue: Queue = queue
        self.kwargs['queue'] = queue
        self.process: Optional[Process] = None
        self.latest_results = None

    def is_alive(self):
        return self.process.is_alive()

    def has_finished(self):
        return not self.process.is_alive()

    def is_suspended(self):
        return self.process is None

    def start(self):
        self.process = Process(target=self.task, kwargs=self.kwargs)
        self.process.start()

    def stop(self, safe=False):
        self.process.terminate()
        if safe:
            self.process.join()
        self.process = None

    def get_last_results(self):
        return self.latest_results

    def receive(self):
        obs = []
        while True:
            try:
                obs.append(self.queue.get(True, timeout=0.01))
            except EmptyQueueException:
                if len(obs) == 0:
                    return None

                self.latest_results = tuple(obs)
                return tuple(obs)


def _slow_function_service(slow_fun, in_queue: Queue, out_queue: Queue):
    call_id = 0

    while True:
        call_id += 1
        kwargs = in_queue.get(block=True)
        result = slow_fun(**kwargs)
        out_queue.put(result)


class AbstractDispatcher:
    """ Manage a series of HPO trials, it can create, suspend and resume those trials
        in function of the results it is receiving through time
     """

    def __init__(self, task, workers):
        """

        :param hpo_algo: Hyper parameter Optimizer, suggest parameters to try and observe current results
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

    def _shutdown(self, gracefull=True):
        # Close self.param_service
        self.param_service.terminate()

        for trial in self.running_trials:
            if gracefull:
                trial.process.join()
            else:
                trial.stop()

        self.manager.shutdown()

    def _suggest(self):
        self.param_input_queue.put({})

    def start_new_trials(self):
        """ check if some params are available in the queue, and create a trial if so """
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

    def receive_and_suspend(self):
        """ iterates through running trials removing finished ones and
            checks if the running trials should be suspended  """
        to_be_suspended = set()
        is_finished = set()

        for trial in self.running_trials:
            result = trial.receive()

            if result is not None and trial.is_alive():
                if self.should_suspend(trial, result):
                    to_be_suspended.add(trial)

            elif not trial.is_alive():
                is_finished.add(trial)

        for trial in to_be_suspended:
            trial.stop()
            self.suspended_trials.add(trial)
            self.running_trials.discard(trial)

        for trial in is_finished:
            self.observe(trial, trial.get_last_results())
            self.finished_trials.add(trial)
            self.running_trials.discard(trial)

        return len(to_be_suspended)

    def resume_pending(self, count):
        """ iterates through the suspended trials and check if they should be resumed """

        to_be_resumed = set()

        for trial in self.suspended_trials:
            if self.should_resume(trial, trial.get_last_results()):
                to_be_resumed.add(trial)

                if len(to_be_resumed) == count:
                    break

        for trial in to_be_resumed:
            self.suspended_trials.discard(trial)
            trial.start()
            self.running_trials.add(trial)

        return len(to_be_resumed)

    def should_resume(self, trial, result):
        return False

    def should_suspend(self, trial, result):
        return False

    def suggest(self):
        raise NotImplementedError()

    def observe(self, trial, result):
        raise NotImplementedError()

    def is_completed(self):
        raise NotImplementedError()


if __name__ == '__main__':
    import time

    def sample_params(hpo_algo, obs, seed):
        conf = copy.deepcopy(hpo_algo)
        conf.observe(obs)

        with Chrono('hpo_time') as timer:
            hpo_algo.get_parans(seed=seed)
            params = conf.get_params(seed=seed)

        params['hpo_time'] = timer
        return params


    class DefaultDispatcher(AbstractDispatcher):
        suggest_id = 0
        observe_id = 0

        def suggest(self):
            with Chrono('hpo_time') as timer:
                self.suggest_id += 1
                print(f'Suggesting {self.suggest_id}')
                time.sleep(1)
            return {'suggest_random': self.suggest_id, 'hpo_time': timer.val}

        def observe(self, trial, result):
            self.observe_id += 1
            print(f'Observing: {self.observe_id} {result}')

        def is_completed(self):
            return len(self.finished_trials) >= 10

        def should_suspend(self, trial, result):
            print(f'Should suspend {result}')
            return False

        def should_resume(self, trial, result):
            return False


    def my_task(suggest_random, queue, hpo_time):
        print(f'Working on {suggest_random}')
        queue.put({
            'params': (('arg1', 0), ('arg2', 0)),
            'objective': 100,
            'hpo_time': hpo_time,
            'exec_time': 5
        })

        time.sleep(5)

        queue.put({
            'params': (('arg1', 0), ('arg2', 0)),
            'objective': 100,
            'hpo_time': hpo_time,
            'exec_time': 10
        })

        time.sleep(5)


    dispatcher = DefaultDispatcher(my_task, 10)
    dispatcher.run()

    for trial in dispatcher.finished_trials:
        print(f'Trial: {trial.kwargs}')
        for result in trial.get_last_results():
            print(f'    - {result}')













