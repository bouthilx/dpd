from collections import namedtuple
import functools
import itertools
import logging

from typing import Iterable
from orion.core.io.space_builder import Space, DimensionBuilder

try:
    import cocoex
except ImportError:
    cocoex = None

try:
    import mahler.client as mahler
except ImportError:
    mahler = None


logger = logging.getLogger(__name__)

if cocoex:
    suite = cocoex.Suite("bbob", "year: 2016", "")


class COCOBenchmark:
    name = 'coco'
    attributes = ['problem_ids', 'dimensions', 'instances']

    def __init__(self, problem_ids=None, instances=None, dimensions=None):
        self.problem_ids = problem_ids if problem_ids else list(range(1, 25))
        self.instances = instances if instances else list(range(1, 6))
        self.dimensions = dimensions if dimensions else suite.dimensions

    def add_subparser(self, parser):
        benchmark_parser = parser.add_parser(self.name)

        for name in self.attributes:
            kwargs = {}
            value = getattr(self, name)

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

        configs = itertools.product(*[getattr(self, name) for name in prod_attributes])

        for config in configs:
            problem = self.build(*config)
            if problem:
                yield problem

    def build(self, problem_id, dimension, instance):
        fixed_attributes = []
        benchmark_config = dict(getattr(self, name) for name in fixed_attributes)
        ProblemType = namedtuple('COCOProblem', ['problem_id', 'dimension', 'instance', 'tags',
                                                 'run', 'space', 'config'])
        try:
            # function, dimension, instance, observer=None
            problem = suite.get_problem_by_function_dimension_instance(
                function=problem_id,
                dimension=dimension,
                instance=instance
            )
        except cocoex.exceptions.NoSuchProblemException:
            return None

        # TODO: inspect build_problem arguments to automatically map with problem_config
        problem_config = dict(problem_id=problem_id, dimension=dimension, instance=instance)
        benchmark_config.update(problem_config)
        benchmark_config['tags'] = create_tags(**problem_config)
        benchmark_config['run'] = functools.partial(coco_run, problem_config=problem_config)
        benchmark_config['space'] = build_space(problem)

        return ProblemType(config=problem_config, **benchmark_config)


def coco_run(problem_config, callback=None, **params):
    problem = build_problem(**problem_config)
    objective = problem([params[get_dim_name(dim)] for dim in range(problem.dimension)])

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


# if mahler is not None:
#    hpo_coco = mahler.operator(resources={'cpu': 2, 'mem': '20MB'}, resumable=False)(hpo_coco)


if cocoex is not None:
    def build(**kwargs):
        config = dict()
        for key in COCOBenchmark.attributes:
            if key in kwargs:
                config[key] = kwargs[key]

        return COCOBenchmark(**config)
