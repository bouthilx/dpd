import logging

import numpy

from repro.hpo.dispatcher.dispatcher import HPODispatcher


logger = logging.getLogger(__name__)


# # NOTE: task name automatically logged in tags
# mahler.register(task, after=None, before=None, tags=())
# mahler.task_count(tags, status=None)
# mahler.find(tags, status=None)


def build(space, configurator_config, max_epochs, grace_period, reduction_factor, brackets,
          max_trials, seed):
    return ASHA(space, configurator_config, max_epochs, grace_period, reduction_factor, brackets,
                max_trials, seed)


class ASHA(HPODispatcher):
    def __init__(self, space, configurator_config, max_epochs, grace_period, reduction_factor,
                 brackets, max_trials, seed):
        super(ASHA, self).__init__(space, configurator_config, max_trials, seed)

        self.max_epochs = max_epochs
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor
        self.trial_info = {}  # Stores Trial -> Bracket

        # Tracks state for new trial add
        self.brackets = [
            _Bracket(self, grace_period, max_epochs, reduction_factor, s)
            for s in range(brackets)
        ]

        logger.info(f'brackets: {[bracket.rungs for bracket in self.brackets]}')

    def make_suggest_parameters(self):
        kwargs = super(ASHA, self).make_suggest_parameters()
        trial_id = kwargs['trial_id']

        sizes = numpy.array([len(b.rungs) for b in self.brackets])
        probs = numpy.e**(sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = numpy.random.choice(len(self.brackets), p=normalized)
        self.trial_info[trial_id] = self.brackets[idx]
        self.brackets[idx].register(trial_id)

        return kwargs

    def _observe(self, *args, **kwargs):
        super(ASHA, self)._observe(*args, **kwargs)

        for bracket in self.brackets:
            bracket.update_rungs()

    def should_suspend(self, trial_id):
        return self.trial_info[trial_id].should_suspend(trial_id)

    def should_resume(self, trial_id):
        return not self.should_suspend(trial_id)

    def is_completed(self):
        return all(bracket.is_completed() for bracket in self.brackets)


class _Bracket():
    def __init__(self, asha, min_t, max_t, reduction_factor, s):
        self.asha = asha
        self.reduction_factor = reduction_factor
        max_rungs = int(numpy.log(max_t / min_t) / numpy.log(reduction_factor) - s + 1)
        self.rungs = [(min(min_t * reduction_factor**(k + s), max_t), set())
                      for k in reversed(range(max_rungs + 1))]
        if len(self.rungs) > 1 and self.rungs[0][0] == self.rungs[1][0]:
            del self.rungs[0]

        logger.info(f'rungs: {self.rungs}')

    def cutoff(self, recorded):
        if not recorded:
            return None
        return numpy.percentile(list(recorded.values()), (1 - 1 / self.rf) * 100)

    def get_rung_id(self, trial_id):
        for rung_id, (_, rung) in enumerate(self.rungs):
            if trial_id in rung:
                return rung_id

        return None

    def should_suspend(self, trial_id):
        rung_id = self.get_rung_id(trial_id)
        last_step = max(self.asha.observations[trial_id].keys())
        return self.rungs[rung_id][0] <= last_step

    def register(self, trial_id):
        self.rungs[-1][1].add(trial_id)

    def get_candidate(self, rung_id):
        budget, rung = self.rungs[rung_id]
        next_rung = self.rungs[rung_id - 1][1]
        completed_trials = []
        for trial_id in rung:
            last_step, objective = self.asha.get_objective(trial_id)
            if last_step and last_step >= budget:
                completed_trials.append((objective, trial_id))

        k = len(completed_trials) // self.reduction_factor
        completed_trials = list(sorted(completed_trials))
        i = 0
        k = min(k, len(completed_trials))
        while i < k:
            objective, trial_id = completed_trials[i]
            if trial_id not in next_rung:
                return trial_id, objective
            i += 1

        return None, None

    def is_completed(self):
        budget, trials = self.rungs[0]
        for trial_id in trials:
            last_step, _ = self.asha.get_objective(trial_id)
            if last_step >= budget:
                return True

        return False

    def update_rungs(self):
        """

        Notes
        -----
            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.
        """
        # NOTE: There should be base + 1 rungs
        for rung_id in range(1, len(self.rungs)):
            candidate, objective = self.get_candidate(rung_id)

            if candidate:
                logger.info(f'{candidate} promoting to {rung_id - 1} {objective}')
                self.rungs[rung_id - 1][1].add(candidate)
                logger.info(
                    'Current rungs: ' +
                    ' '.join('{}:{}'.format(rung_b, len(rung))
                             for rung_b, rung in self.rungs))
                return
