import time

from repro.utils.chrono import Chrono
from repro.hpo.dispatcher import AbstractDispatcher


class TestDispatcher(AbstractDispatcher):
    suggest_id = 0
    observe_id = 0
    observed = []

    def suggest(self):
        with Chrono('hpo_time') as timer:
            print(f'Suggesting {self.suggest_id}')
            self.suggest_id += 1
            time.sleep(2)
        return {'suggest_random': self.suggest_id, 'hpo_time': timer.val}

    def observe(self, trial, result):
        self.observe_id += 1
        self.observed.append(result[-1].get('objective'))
        self.suggest_id += result[-1].get('objective')
        print(f'Observing: {self.observe_id} {result}')

    def is_completed(self):
        return len(self.finished_trials) >= 10

    def should_suspend(self, trial):
        print(f'Should suspend')
        return False

    def should_resume(self, trial):
        return False


def test_task(suggest_random, queue, hpo_time):
    print(f'Working on {suggest_random}')
    queue.put({
        'params': (('arg1', 0), ('arg2', 0)),
        'objective': suggest_random,
        'hpo_time': hpo_time,
        'exec_time': 5,
        'finished': False
    })

    time.sleep(1)
    suggest_random += 1

    queue.put({
        'params': (('arg1', 0), ('arg2', 0)),
        'objective': suggest_random,
        'hpo_time': hpo_time,
        'exec_time': 10,
        'finished': True
    })


def test_dispatcher():
    max_tasks = 10

    dispatcher = TestDispatcher(test_task, max_tasks)
    dispatcher.run()
    assert dispatcher.observed[-1] == max_tasks + 1


if __name__ == '__main__':
    test_dispatcher()

