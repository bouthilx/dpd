import inspect
import copy
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
        return {'id', 'params', 'latest_results', 'timestamps', 'results'}


class ResumeManagerAspect(ResumableAspect):
    def state_attributes(self):
        return {'running_trials', 'suspended_trials', 'finished_trials',
                'dispatcher', 'pending_params', 'resource_manager', 'trials'}

    def resume(self, obj: 'HPOManager', state: Dict[str, any]):
        from repro.hpo.trial.builtin import Trial

        # TODO: Restore using the proper backend
        def make_trial(trial_state, queue):
            t = Trial(trial_state['id'], obj.task, trial_state['params'], queue)
            t.latest_results = trial_state['latest_results']
            return t

        obj.dispatcher = resume(obj.dispatcher, state['dispatcher'])
        obj.resource_manager = resume(obj.resource_manager, state['resource_manager'])

        trials = obj.trials
        obj.trials = []
        for trial_state in trials:
            obj._insert_trial(make_trial(trial_state, obj.manager.Queue()))

        for trial_id in obj.running_trials:
            trial = obj.get_trial(trial_id)
            trial.start()
            obj.running_trials.add(trial_id)

        # Compute the trial_count remaining
        obj.trial_count = max(len(obj.finished_trials) + len(obj.suspended_trials) + len(obj.running_trials) - 1, 0)
        obj.dispatcher.trial_count = obj.trial_count
        return obj


def _to_defaultdict(diction):
    default = defaultdict(dict)

    for hex, steps in diction.items():
        for k, v in steps.items():
            default[hex][int(k)] = v

    return default


class ResumeDispatcherAspect(ResumableAspect):
    def state_attributes(self):
        return {'seeds', 'observations', 'buffered_observations', 'finished', 'params'}

    def resume(self, obj: 'HPODispatcher', state: Dict[str, any]):
        super(ResumeDispatcherAspect, self).resume(obj, state)
        # self.observations: Dict[str, Dict[str, int]]

        obj.observations = _to_defaultdict(obj.observations)

        obj.finished = set(obj.finished)
        return obj


class ResumeASHAAspect(ResumeDispatcherAspect):
    def state_attributes(self):
        return super(ResumeASHAAspect, self).state_attributes() & {'rungs'}

    def resume(self, obj: 'ASHA', state: Dict[str, any]):
        super(ResumeASHAAspect, self).resume(obj, state)
        # self.observations: Dict[str, Dict[str, int]]

        obj.rungs = defaultdict(set)
        obj.rungs.update({k: set(v) for k, v in state['rungs'].items()})

        return obj


class ResumeResourceManagerAspect(ResumableAspect):
    def state_attributes(self):
        return {'id'}


class ResumeNdArray(ResumableAspect):
    def state(self, obj) -> any:
        return obj.tolist()


class ResumeNpInt(ResumableAspect):
    def state(self, obj) -> any:
        return int(obj)


class ResumeSet(ResumableAspect):
    def state(self, obj) -> any:
        return [state(i) for i in obj]


class ResumeDict(ResumableAspect):
    def state(self, obj) -> any:
        return {k: state(v) for k, v in obj.items()}


class ResumeList(ResumableAspect):
    def state(self, obj) -> any:
        return [state(i) for i in obj]


def _register():
    if set not in _resumable_aspect:
        # TODO: Verify that different backends are supported properly
        from repro.hpo.trial.builtin import Trial
        from repro.hpo.resource.builtin import ResourceManager
        from repro.hpo.dispatcher.dispatcher import HPODispatcher
        from repro.hpo.dispatcher.asha import ASHA
        from repro.hpo.manager import HPOManager

        ResumableAspect.register(ResumeSet(), set)
        ResumableAspect.register(ResumeDict(), dict)
        ResumableAspect.register(ResumeList(), list)
        ResumableAspect.register(ResumeNpInt(), numpy.int32)
        ResumableAspect.register(ResumeNpInt(), numpy.int64)
        ResumableAspect.register(ResumeNdArray(), numpy.ndarray)
        ResumableAspect.register(ResumeDispatcherAspect(), HPODispatcher)
        ResumableAspect.register(ResumeASHAAspect(), ASHA)
        ResumableAspect.register(ResumeResourceManagerAspect(), ResourceManager)
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
