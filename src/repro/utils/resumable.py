from typing import Dict
import numpy as np


class Resumable:
    __state_attributes__ = None


def state(obj: Resumable) -> any:
    if hasattr(obj, 'state'):
        return obj.state()

    if hasattr(obj, '__state_attributes__'):
        state_dict = {}
        for attr in obj.__state_attributes__:
            state_dict[attr] = state(getattr(obj, attr))

        return state_dict

    # numpy
    if hasattr(obj, 'tolist'):
        obj = obj.tolist()

    if isinstance(obj, list) or isinstance(obj, set):
        return [state(i) for i in obj]

    return obj


def resume(obj: Resumable, state: Dict[str, any], default=False) -> Resumable:
    if not default and hasattr(obj, 'resume'):
        return obj.resume(state)

    if hasattr(obj, '__state_attributes__'):
        for attr in obj.__state_attributes__:
            setattr(obj, attr, state[attr])

        return obj

    raise RuntimeError('Resumable interface not implemented')
