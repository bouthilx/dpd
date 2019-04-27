from typing import List, Tuple, Dict, Optional, Iterable
from dataclasses import dataclass, field
from collections import namedtuple
import itertools

import cocoex
import copy


default_scenarios = [
    '0.0',  # No diff
    '2.1',  # Fewer H-Ps
    '2.2',  # More H-Ps
    '2.3.a',  # Prior changed
    '2.3.b',  # Prior changed
    '2.4.a',
    '2.4.b',
    '3.1',  # Code change without any effect
    '3.2',  # Loss is reversed
    '3.3',  # Loss is scaled
    '3.4',  # H-P is reversed
    '3.5'
]


@dataclass
class CocoBenchmark:
    name: str = 'coco'
    problem_ids: List[int] = field(default_factory=lambda: list(range(1, 25)))
    instances: List[int] = field(default_factory=lambda: list(range(1, 6)))
    scenarios: List[str] = field(default_factory=lambda: copy.deepcopy(default_scenarios))
    workers: List[int] = field(default_factory=lambda: list(range(1, 32)))
    previous_tags: Optional[List[str]] = field(default_factory=list)
    warm_start = 0
    suite = cocoex.Suite("bbob", "year: 2016", "")
    dimensions: List[int] = field(default_factory=lambda: cocoex.Suite("bbob", "year: 2016", "").dimensions)

    @staticmethod
    def atttributes() -> Iterable[str]:
        # the positions of attributes matters as they will initialize the CocoProblem tuple
        return ['problem_ids', 'dimensions', 'instances', 'scenarios', 'workers', 'previous_tags', 'warm_start']


def add_subparser(parser, benchmark: CocoBenchmark):
    benchmark_parser = parser.add_parser(benchmark.name)

    for name in benchmark.atttributes():
        kwargs = {}
        value = getattr(benchmark, name)

        if isinstance(value, list):
            kwargs['choices'] = value
            kwargs['nargs'] = '*'
            if value:
                kwargs['type'] = type(value[0])
        else:
            kwargs['default'] = value
            kwargs['type'] = type(value)

        benchmark_parser.add_argument(f'--{name}', **kwargs)

    return benchmark_parser


def coco_problems(benchmark: CocoBenchmark) -> Iterable[any]:
    ProblemType = namedtuple('CocoProblem', benchmark.atttributes())
    prod_attributes = ['problem_ids', 'dimensions', 'instances', 'scenarios', 'workers']
    fixed_attributes = ['previous_tags', 'warm_start']

    configs = itertools.product(*[getattr(benchmark, name) for name in prod_attributes])
    fixed_arguments = tuple([getattr(benchmark, name) for name in fixed_attributes])

    for config in configs:
        try:
            # function, dimension, instance, observer=None
            benchmark.suite.get_problem_by_function_dimension_instance(
                function=config[0],
                dimension=config[1],
                instance=config[2]
            )
        except cocoex.exceptions.NoSuchProblemException:
            continue

        yield ProblemType(*(config + fixed_arguments))


if __name__ == '__main__':
    import argparse

    bench = CocoBenchmark(problem_ids=[1, 2, 3])

    parser = argparse.ArgumentParser('testing')
    subparser = parser.add_subparsers()

    subparser = add_subparser(subparser, bench)

    print('---')
    for probs in coco_problems(bench):
        print(probs)

    print('---')
    parser.parse_args()
