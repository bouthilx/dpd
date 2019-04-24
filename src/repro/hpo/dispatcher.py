
# A trial is either a (python) process or a task. It is statefull and can be resumed.

class Suspender:
    def __init__(self):
        pass

    def update_trials(self, trials):
        while not self.queue.empty():
            try:
                trials.append(self.queue.get(timeout=0.001))
            except queue.Empty:
                break

    def run(self):
        trials = []
        while True:
            self.update_trials(trials)

            for trial in trials:
                dispatcher.monitor(trial)


class AbstractDispatcher:
    def __init__(self, algoritm):
        self.algoritm = algoritm
        self.fork = None

    def observe(self, points):
        pass

    def create_trial(self, params):
        # Callback is sending result back to the trial object (db in mahler, queue? in pypool)
        (self.fct, params, callback=self.signal_step)

    def should_suspend(self):
        raise NotImplementedError

    def monitor(self, trial):
        if self.should_suspend(trial):
            trial.kill()

    def suggest(self):
        resume = self.suggest_resume()
        if resume:
            trial = resume
        else:
            trial = self.suggest_algo()

        # We need to fork a process that will answer callbacks in parallel
        self.fork.queue.put(trial)

        return trial

    def suggest_resume(self):
        pass

    def suggest_algo(self):
        algo = copy.deepcopy(self.algoritm)
        algo.observe(self.uncompleted_trials)

        return algo.suggest()

    def is_completed(self):
        return False


class Pool:

    def add(self, process):
        self.processes.append(process)

    def wait(self, num_workers=1):
        completed = []
        while len(completed) < num_workers:
            for process in self.processes:
                if not process.is_alive():
                    completed.append(process)
                    self.processes.remove(process)

        return completed


def main():

    dispatcher = build_dispatcher(fct, algo)
    
    while not dispatcher.is_completed():

        while pool.size() < num_workers:
            trial = dispatcher.suggest()  # This may take a while
            pool.add(trial)

        pool.wait(1)
























