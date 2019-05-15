from typing import Callable, Dict, List       

from hpo.dispatcher.dispatcher import HPODispatcher



class MedianStoppingRule(HPODispatcher):

    def _compute_median(self, step):
        values = []
        for trial_objectives in self.observations.values():
            if last_step in trial_objectives:
                values.append(trial_objectives[last_step])

        return numpy.median(values)

    def should_suspend(self, trial_id: str) -> bool:
        
        last_step, objective = self.get_objective(trial_id)

        return objective is not None and objective > self._compute_median(last_step)

    def should_resume(self, trial_id) -> bool:
        return not self.should_suspend(trial_id)

    def is_completed(self):
        pass

    def _observe(self, trial_id, step, objective, **kwargs):
        self.observations[trial_id][step] = objective
