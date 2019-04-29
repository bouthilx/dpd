import time

from repro.utils.chrono import Chrono
from repro.hpo.dispatcher.dispatcher import HPOManager, HPODispatcher


class TestDispatcher(HPODispatcher):
    suggest_id = 0
    observe_id = 0
    observed = []

    def suggest(self, seed):
        with Chrono('hpo_time') as timer:
            print(f'Suggesting {self.suggest_id}')
            self.suggest_id += 1
            time.sleep(1)
        return {'suggest_random': self.suggest_id, 'hpo_time': timer.val}

    def observe(self, trial, result):
        self.observe_id += 1
        self.observed.append(result[-1].get('objective'))
        self.suggest_id += result[-1].get('objective')
        print(f'Observing: {self.observe_id} {result}')
        super(TestDispatcher, self).observe(trial, result)

    def is_completed(self):
        return len(self.finished) >= 10

    def should_suspend(self, trial):
        # print(f'Should suspend')
        return False

    def should_resume(self, trial):
        return False


def test_task(suggest_random, hpo_time, callback):
    print(f'Working on {suggest_random}')

    callback(**{
        'params': (('arg1', 0), ('arg2', 0)),
        'objective': suggest_random,
        'hpo_time': hpo_time,
        'exec_time': 5,
        'step': 1,
        'finished': False
    })

    time.sleep(10)
    suggest_random += 1

    callback(**{
        'params': (('arg1', 0), ('arg2', 0)),
        'objective': suggest_random,
        'hpo_time': hpo_time,
        'exec_time': 10,
        'step': 2,
        'finished': True
    })


def test_dispatcher():
    max_tasks = 10

    dispatcher = TestDispatcher(None, dict(name='random_search'), max_trials=max_tasks, seed=1)
    manager = HPOManager(dispatcher, test_task, max_trials=max_tasks, workers=5)
    manager.run()
    print(manager.trial_count)
    print(dispatcher.finished)
    print(len(dispatcher.finished))
    assert dispatcher.observed[-1] == max_tasks + 1


if __name__ == '__main__':
    test_dispatcher()
