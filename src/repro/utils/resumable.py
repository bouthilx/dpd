import inspect
import numpy
from collections import defaultdict
from typing import Dict, Optional, Set

_resumable_aspect = {}


class ResumableAspect:
    """
        > aspect-oriented programming (AOP) is a programming paradigm that aims to increase
        > modularity by allowing the separation of cross-cutting concerns.
        > It does so by adding additional behavior to existing code (an advice) without modifying
        > the code itself.

        Here we implement all the necessary tools to resume HPOManager, Trial and HPODispatcher
        without modifying them. The needs to be registered at the end.

        Resume Aspect implements two methods `state` which serialize a given object into a dict and
        a `resume` that initialize a given object to the specified state
    """
    @staticmethod
    def register(impl: 'ResumableAspect', cls):
        _resumable_aspect[cls] = impl

    def state_attributes(self) -> Optional[Set[str]]:
        return None

    def state(self, obj) -> any:
        attrs = self.state_attributes()

        if attrs is not None:
            state_dict = {}
            for attr in attrs:
                state_dict[attr] = state(getattr(obj, attr))

            return state_dict

        return obj

    def resume(self, obj, state):
        attrs = self.state_attributes()

        if attrs is not None:
            for attr in attrs:
                setattr(obj, attr, state[attr])

            return obj

        raise RuntimeError('Resumable aspect not implemented')


class ResumeTrialAspect(ResumableAspect):
    def state_attributes(self):
        return {'id', 'params', 'latest_results'}


class ResumeManagerAspect(ResumableAspect):
    def state_attributes(self):
        return {'trial_count', 'running_trials', 'suspended_trials', 'finished_trials',
                'dispatcher'}

    def resume(self, obj: 'HPOManager', state: Dict[str, any]):
        from repro.hpo.trial.builtin import Trial

        # TODO: Restore using the proper backend
        def make_trial(trial_state, queue):
            t = Trial(trial_state['id'], obj.task, trial_state['params'], queue)
            t.latest_results = trial_state['latest_results']
            return t

        obj.dispatcher = resume(obj.dispatcher, state['dispatcher'])
        obj.trial_count = state['trial_count']

        obj.running_trials = {make_trial(t, obj.manager.Queue()) for t in state['running_trials']}
        obj.suspended_trials = {make_trial(t, obj.manager.Queue())
                                for t in state['suspended_trials']}
        obj.finished_trials = {make_trial(t, None) for t in state['finished_trials']}

        for trial in obj.running_trials:
            trial.start()

        return obj


class ResumeDispatcherAspect(ResumableAspect):
    def state_attributes(self):
        return {'trial_count', 'seeds', 'observations', 'buffered_observations', 'finished',
                'params'}

    def resume(self, obj: 'HPODispatcher', state: Dict[str, any]):
        super(ResumeDispatcherAspect, self).resume(obj, state)
        # self.observations: Dict[str, Dict[str, int]]
        obs = obj.observations
        obj.observations = defaultdict(dict)

        for hex, steps in obs.items():
            for k, objective in steps.items():
                obj.observations[hex][int(k)] = objective

        obj.finished = set(obj.finished)
        return obj


class ResumeNdArray(ResumableAspect):
    def state(self, obj) -> any:
        return obj.tolist()


class ResumeSet(ResumableAspect):
    def state(self, obj) -> any:
        return [state(i) for i in obj]


class ResumeList(ResumableAspect):
    def state(self, obj) -> any:
        return [state(i) for i in obj]


def _register():
    if set not in _resumable_aspect:
        # TODO: Verify that different backends are supported properly
        from repro.hpo.trial.builtin import Trial
        from repro.hpo.dispatcher.dispatcher import HPODispatcher
        from repro.hpo.manager import HPOManager

        ResumableAspect.register(ResumeSet(), set)
        ResumableAspect.register(ResumeList(), list)
        ResumableAspect.register(ResumeNdArray(), numpy.ndarray)
        ResumableAspect.register(ResumeDispatcherAspect(), HPODispatcher)
        ResumableAspect.register(ResumeManagerAspect(), HPOManager)
        ResumableAspect.register(ResumeTrialAspect(), Trial)


_register()


def state(obj: any) -> any:
    classes = inspect.getmro(obj.__class__)

    # classes is in resolution order, so you can register specialized version for your class
    for cls in classes:
        if cls in _resumable_aspect:
            return _resumable_aspect[cls].state(obj)

    # lots of object in python are just dicts
    return obj


def resume(obj: any, state: any) -> any:
    classes = inspect.getmro(obj.__class__)

    for cls in classes:
        if cls in _resumable_aspect:
            return _resumable_aspect[cls].resume(obj, state)

    raise RuntimeError(f'Resumable aspect not implemented for {obj.__class__}')


def is_resumeable(obj: any) -> bool:
    classes = inspect.getmro(obj.__class__)

    for cls in classes:
        if cls in _resumable_aspect:
            return True

    return False
