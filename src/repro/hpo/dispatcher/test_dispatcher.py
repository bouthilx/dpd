import time

from repro.utils.chrono import Chrono
from repro.hpo.dispatcher.dispatcher import HPOManager, HPODispatcher
from repro.utils.checkpoint import resume_from_checkpoint

mock_sleep = 10
suggest_sleep = 1


class MockDispatcher(HPODispatcher):
    suggest_id = 0
    observe_id = 0
    observed = []

    def __init__(self, max_trials):
        super(MockDispatcher, self).__init__(None, {}, max_trials)

    def suggest(self, seed):
        with Chrono('hpo_time') as timer:
            print(f'Suggesting {self.suggest_id}')
            self.suggest_id += 1
            time.sleep(suggest_sleep)
        return {'suggest_random': self.suggest_id, 'hpo_time': timer.val}

    def observe(self, trial, result):
        self.observe_id += 1
        self.observed.append(result[-1].get('objective'))
        self.suggest_id += result[-1].get('objective')
        print(f'Observing: {self.observe_id} {result}')

        if trial.has_finished():
            self.finished.add(trial.id)

        super(MockDispatcher, self).observe(trial, result)

    def is_completed(self):
        # print(len(self.finished), self.max_trials)
        return len(self.finished) >= self.max_trials

    def should_suspend(self, trial):
        # print(f'Should suspend')
        return False

    def should_resume(self, trial):
        return False


def mock_task(suggest_random, hpo_time, callback):
    print(f'Working on {suggest_random}')

    callback(**{
        'params': (('arg1', 0), ('arg2', 0)),
        'objective': suggest_random,
        'hpo_time': hpo_time,
        'exec_time': 5,
        'step': 1,
        'finished': False
    })

    time.sleep(mock_sleep)
    suggest_random += 1

    callback(**{
        'params': (('arg1', 0), ('arg2', 0)),
        'objective': suggest_random,
        'hpo_time': hpo_time,
        'exec_time': 10,
        'step': 2,
        'finished': True
    })


def run_mock_dispatcher():
    global mock_sleep, suggest_sleep
    mock_sleep = 1
    suggest_sleep = 1
    max_tasks = 10

    dispatcher = MockDispatcher(max_tasks)
    manager = HPOManager(dispatcher, mock_task, max_trials=max_tasks, workers=5)
    manager.run()

    print(manager.trial_count)
    print(dispatcher.finished)
    print(len(dispatcher.finished))

    assert dispatcher.observed[-1] == max_tasks + 1


def run_mock_suspend_resume():
    import numpy

    max_tasks_real = 10
    seeds = numpy.random.RandomState(0).randint(0, 100000, size=(max_tasks_real,))

    def start_and_suspend():
        max_tasks = 5
        dispatcher = MockDispatcher(max_tasks)
        dispatcher.seeds = seeds

        manager = HPOManager(dispatcher, mock_task, max_trials=max_tasks, workers=5)
        manager.enable_checkpoints(
            name='test_hpo',
            every=1,
            archive_folder='/tmp/checkpoints/repro/'
        )
        manager.run()

    def resume_and_finish():
        max_tasks = max_tasks_real

        dispatcher = MockDispatcher(max_tasks)
        manager = HPOManager(dispatcher, mock_task, max_trials=max_tasks, workers=5)

        manager = resume_from_checkpoint(manager, '/tmp/checkpoints/repro/test_hpo.json')
        manager.run()

        return dispatcher

    start_and_suspend()

    dispatcher = resume_and_finish()


if __name__ == '__main__':
    # run_mock_dispatcher()
    run_mock_suspend_resume()
