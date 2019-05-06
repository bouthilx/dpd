import copy
from collections import defaultdict
import logging
import uuid
import numpy

from typing import Dict, Tuple

from repro.hpo.configurator.base import build_configurator


logger = logging.getLogger(__name__)


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
