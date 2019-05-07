from collections import namedtuple
from typing import Iterable
import pickle
import time

import numpy


try:
    import mahler.client as mahler
except ImportError:
    mahler = None

from orion.core.io.space_builder import Space, DimensionBuilder


T = 100


class CurveBenchmark:
    name = 'curves'

    def add_subparser(self, parser):
        return parser.add_parser(self.name)

    @property
    def problems(self) -> Iterable[any]:
        yield self.build()

    def build(self):
        ProblemType = namedtuple('CurveProblem', ['tags', 'run', 'space', 'config'])

        problem_config = dict(
            config={},
            tags=create_tags(),
            run=curve_run,
            space=build_space())

        return ProblemType(**problem_config)


def build_space():
    space = Space()
    space['a'] = DimensionBuilder().build('a', 'uniform(0.1, 2.0)')
    space['b'] = DimensionBuilder().build('b', 'uniform(0, 10.0)')

    return space


def create_tags():
    return ['b-curves']


def forget_curve(t, a=1.84, b=1.25, c=1.84):
    o = a / (numpy.log(t) ** b + c)
    return (o <= 1) * o + (o > 1)


def gated(t, a, b):
    return forget_curve(t, a=max(a, b), b=a, c=a)


def curve_run(a, b, callback):

    if callback and not hasattr(callback, '__call__'):
        callback = pickle.loads(callback)

    objectives = []

    for t in range(1, T + 1):
        y = gated(t, a, b)
        time.sleep(0.01)
        if callback:
            callback(step=t, objective=y, finished=(t) >= T)
        objectives.append({'epoch': t, 'objective': y})

    return {'all': objectives}


if mahler:
    resources = {'cpu': 1, 'mem': '1GB', 'usage': {'cpu': {'util': 0, 'memory': 2 ** 30}}}
    curve_run = mahler.operator(resources=resources)(curve_run)


def build(**kwargs):
    return CurveBenchmark()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = numpy.arange(1, 100 + 1)
    plt.plot(x, gated(x, 0.3, 0.1), label='0.1 0.1')
    plt.plot(x, gated(x, 0.3, 0.3), label='0.3 0.3')
    plt.plot(x, gated(x, 0.3, 10), label='0.3 10')
    plt.plot(x, gated(x, 2, 0.3), label='2 0.3')
    plt.plot(x, gated(x, 2, 10), label='2 10')
    plt.legend()
    plt.show()
