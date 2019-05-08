from collections import namedtuple
import copy
import functools
import itertools
import logging

from typing import Iterable
from orion.core.io.space_builder import Space, DimensionBuilder

try:
    import mahler.client as mahler
except ImportError:
    mahler = None

try:
    from repro.train import train
except ImportError:
    train = None

from repro.utils.flatten import flatten, unflatten


logger = logging.getLogger(__name__)


DATASETS = ['mnist', 'fashionmnist', 'svhn', 'cifar10', 'cifar100', 'tinyimagenet']
DATASET_FOLDS = list(range(1, 11))
MODELS = (
    ['lenet'] +
    [f'vgg{i}' for i in [11, 13, 16, 19]] +
    [f'resnet{i}' for i in [18, 34, 50, 101]] +
    [f'preactresnet{i}' for i in [18, 34, 50, 101]] +
    [f'densenet{i}' for i in [121, 161, 169, 201]] +
    ['mobilenetv2'])
OPTIMIZERS = ['sgd', 'adam']


class VisionDLBenchmark:
    name = 'visiondl'
    attributes = ['datasets', 'dataset_folds', 'models', 'optimizers']

    def __init__(self, datasets=None, dataset_folds=None, models=None, optimizers=None):
        self.datasets = datasets if datasets else DATASETS
        self.dataset_folds = dataset_folds if dataset_folds else DATASET_FOLDS
        self.models = models if models else MODELS
        self.optimizers = optimizers if optimizers else OPTIMIZERS

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
        prod_attributes = ['datasets', 'dataset_folds', 'models', 'optimizers']

        configs = itertools.product(*[getattr(self, name) for name in prod_attributes])

        for config in configs:
            problem = self.build(*config)
            if problem:
                yield problem

    def build(self, dataset, dataset_fold, model, optimizer):
        fixed_attributes = []
        benchmark_config = dict(getattr(self, name) for name in fixed_attributes)
        ProblemType = namedtuple('VisionDLProblem', ['dataset', 'dataset_fold', 'model', 'optimizer',
                                                   'tags', 'run', 'space', 'config'])

        # TODO: inspect build_problem arguments to automatically map with problem_config
        problem_config = dict(dataset=dataset, dataset_fold=dataset_fold, model=model,
                              optimizer=optimizer)
        benchmark_config.update(problem_config)
        benchmark_config['tags'] = create_tags(**problem_config)
        benchmark_config['run'] = functools.partial(visiondl_run, problem_config=problem_config)
        benchmark_config['space'] = build_space(model, optimizer)

        return ProblemType(config=problem_config, **benchmark_config)


def visiondl_run(problem_config, callback=None, **params):
    return train(callback=callback, **merge(expand_problem_config(**problem_config), params))


def expand_problem_config(dataset, dataset_fold, model, optimizer):
    return {
        'data': {
            'name': dataset,
            'seed': (0, dataset_fold),
            'batch_size': 128},
        'model': {
            'name': model},
        'optimizer': OPTIMIZER_CONFIGS[optimizer],
        'model_seed': dataset_fold,
        'sampler_seed': dataset_fold
        }


OPTIMIZER_CONFIGS = {
    'sgd': {
        'name': 'sgd',
        'momentum': 0.9,
        'weight_decay': 10e-10},
    'adam': {
        'name': 'adam'}}


OPTIMIZER_SPACES = {
    'sgd': {
        'lr': 'loguniform(1.0e-5, 1.0)',
        'lr_scheduler': {
            'patience': 'loguniform(5, 50, discrete=True)',
            'factor': 'loguniform(0.05, 0.5)'
        },
        'momentum': 'uniform(0., 0.9)',
        'weight_decay': 'loguniform(1.0e-10, 1.0e-3)'
    },
    'adam': {
        'lr': 'loguniform(1.0e-5, 1.0)',
        'betas': 'loguniform(0.7, 1.0, shape=2)',
        'weight_decay': 'loguniform(1.0e-10, 1.0e-3)'
    }
}


def create_tags(dataset, dataset_fold, model, optimizer):
    tags = [
        'b-visiondl',
        f'd-{dataset}',
        f'f-{dataset_fold}',
        f'm-{model}',
        f'o-{optimizer}'
    ]

    return tags


def build_space(model, optimizer, **space_config):

    space = Space()

    dimension_builder = DimensionBuilder()

    full_space_config = {
        'optimizer': OPTIMIZER_SPACES[optimizer]}

    full_space_config['optimizer'].update(space_config.get('optimizer', {}))

    for name, prior in flatten(full_space_config).items():
        if not prior:
            continue
        try:
            space[name] = dimension_builder.build(name, prior)
        except TypeError as e:
            print(str(e))
            print('Ignoring key {} with prior {}'.format(name, prior))

    return space


def merge(config, subconfig):
    flattened_config = copy.deepcopy(flatten(config))
    flattened_config.update(flatten(subconfig))
    return unflatten(flattened_config)


if train is not None:
    def build(datasets=None, dataset_folds=None, models=None, optimizers=None, **kwargs):
        return VisionDLBenchmark(datasets, dataset_folds, models, optimizers)
