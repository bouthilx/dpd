from collections import namedtuple
import functools
import itertools
import logging
import random
import copy
import numpy

from typing import List, Tuple, Dict, Optional, Iterable
from dataclasses import dataclass, field

from orion.core.io.space_builder import Space, DimensionBuilder

from repro.utils.distributed import LazyInstantiator, make_pool
from repro.utils.chrono import Chrono

try:
    import cocoex
except ImportError:
    cocoex = None

try:
    import mahler.client as mahler
    from mahler.core.utils.errors import SignalInterruptTask, SignalSuspend
except ImportError:
    mahler = None


logger = logging.getLogger(__name__)

if cocoex:
    suite = cocoex.Suite("bbob", "year: 2016", "")


class COCOBenchmark:
    name = 'coco'
    attributes =  ['problem_ids', 'dimensions', 'instances']

    def __init__(self, problem_ids=None, instances=None, dimensions=None):
        self.problem_ids = problem_ids if problem_ids else list(range(1, 25))
        self.instances = instances if instances else list(range(1, 6))
        self.dimensions = dimensions if dimensions else suite.dimensions

    def add_subparser(self, parser):
        benchmark_parser = parser.add_parser(self.name)

        for name in self.attributes:
            kwargs = {}
            value = getattr(self, name)

            print(name, value)
            if isinstance(value, list):
                kwargs['choices'] = value
                kwargs['nargs'] = '*'
                if value:
                    kwargs['type'] = type(value[0])
            else:
                kwargs['default'] = value
                kwargs['type'] = type(value)

            benchmark_parser.add_argument('--{}'.format(name.replace('_', '-')), **kwargs)

        return benchmark_parser

    @property
    def problems(self) -> Iterable[any]:
        prod_attributes = ['problem_ids', 'dimensions', 'instances']
        problem_attributes = list(map(lambda s: s.rstrip('s'), prod_attributes))
        ProblemType = namedtuple('COCOProblem', problem_attributes + ['tags', 'run', 'space'])
        fixed_attributes = []

        configs = itertools.product(*[getattr(self, name) for name in prod_attributes])
        benchmark_config = dict(getattr(self, name) for name in fixed_attributes)

        for problem_config in configs:
            print(type(problem_config), problem_config)
            print(benchmark_config)
            try:
                # function, dimension, instance, observer=None
                problem = suite.get_problem_by_function_dimension_instance(
                    function=problem_config[0],
                    dimension=problem_config[1],
                    instance=problem_config[2]
                )
            except cocoex.exceptions.NoSuchProblemException:
                continue
            
            # TODO: inspect build_problem arguments to automatically map with problem_config
            problem_config = dict(zip(problem_attributes, problem_config))
            benchmark_config.update(problem_config)
            benchmark_config['tags'] = create_tags(**problem_config)
            benchmark_config['run'] = functools.partial(coco_run, problem_config=problem_config)
            benchmark_config['space'] = build_space(problem)

            yield ProblemType(**benchmark_config)


def build_space(problem):
    space = Space()
    for dim in range(problem.dimension):
        name = get_dim_name(dim)

        if space_config.get(name, {}) is None:
            continue

        config = dict(
            prior='uniform', lower_bound=problem.lower_bounds[dim],
            upper_bound=problem.upper_bounds[dim])

        space[name] = DimensionBuilder().build(
            name, '{prior}({lower_bound}, {upper_bound})'.format(**config))

    return space


def coco_run(problem_config, callback=None, **params):
    problem = build_problem(**problem_config)
    objective = problem([params[get_dim_name(dim)] for dim in range(problem.dimension)])
    print(params, objective)
    if callback:
        callback(step=1, objective=objective, finished=True)


def create_tags(problem_id, dimension, instance):
    tags = [
        'coco',
        'f{:03d}'.format(problem_id),
        'd{:03d}'.format(dimension),
        'i{:02d}'.format(instance),
    ]

    return tags


DIM_NAME_TEMPLATE = 'dim-{:03d}'


def get_dim_name(dim_id):
    return DIM_NAME_TEMPLATE.format(dim_id)


def build_problem(problem_id, dimension, instance):
    return suite.get_problem_by_function_dimension_instance(problem_id, dimension, instance)


def build_space(problem, **space_config):
    space = Space()
    for dim in range(problem.dimension):
        name = get_dim_name(dim)

        if space_config.get(name, {}) is None:
            continue

        config = dict(
            prior='uniform', lower_bound=problem.lower_bounds[dim],
            upper_bound=problem.upper_bounds[dim])
        config.update(space_config.get(name, {}))

        space[name] = DimensionBuilder().build(
            name, '{prior}({lower_bound}, {upper_bound})'.format(**config))

    return space


#if mahler is not None:
#    hpo_coco = mahler.operator(resources={'cpu': 2, 'mem': '20MB'}, resumable=False)(hpo_coco)


if cocoex is not None:
    def build(**kwargs):
        config = dict()
        for key in COCOBenchmark.attributes:
            if key in kwargs:
                config[key] = kwargs[key]

        return COCOBenchmark(**config)
