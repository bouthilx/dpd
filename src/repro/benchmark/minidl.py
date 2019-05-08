from collections import namedtuple
import copy
import functools
import itertools
import logging
import pickle

from typing import Iterable
from orion.core.io.space_builder import Space, DimensionBuilder

from repro.utils.flatten import flatten, unflatten


logger = logging.getLogger(__name__)


try:
    import mahler.client as mahler
except ImportError as e:
    logger.warning(f'Cannot import mahler: {e}')
    mahler = None

try:
    from repro.train import train
except ImportError as e:
    logger.warning(f'Cannot import train: {e}')
    train = None


DATASETS = ['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'tinyimagenet']
DATASET_FOLDS = list(range(1, 11))
MODELS = ['logreg', 'small_mlp', 'large_mlp', 'small_conv', 'large_conv']
OPTIMIZERS = ['sgd', 'adam']


class MiniDLBenchmark:
    name = 'minidl'
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
        ProblemType = namedtuple('MiniDLProblem', ['dataset', 'dataset_fold', 'model', 'optimizer',
                                                   'tags', 'run', 'space', 'config'])

        # TODO: inspect build_problem arguments to automatically map with problem_config
        problem_config = dict(dataset=dataset, dataset_fold=dataset_fold, model=model,
                              optimizer=optimizer)
        benchmark_config.update(problem_config)
        benchmark_config['tags'] = create_tags(**problem_config)
        benchmark_config['run'] = get_function(problem_config)
        benchmark_config['space'] = build_space(model, optimizer)

        return ProblemType(config=problem_config, **benchmark_config)


def minidl_run(problem_config, model=None, optimizer=None, callback=None):
    config = expand_problem_config(**problem_config)

    params = dict()
    if model:
        params['model'] = model
    if optimizer:
        params['optimizer'] = optimizer

    config = merge(config, params)

    if callback and not hasattr(callback, '__call__'):
        callback = pickle.loads(callback)

    return train(callback=callback, **config)


def expand_problem_config(dataset, dataset_fold, model, optimizer):
    return {
        'data': {
            'name': dataset,
            'seed': (0, dataset_fold),
            'mini': True,
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


MODEL_SPACES = {
    'logreg': {},
    'small_mlp': {
        'width': 'loguniform(10, 1000, discrete=True)',
    },
    'large_mlp': {
        'width': 'loguniform(10, 1000, discrete=True)',
        'reductions': 'uniform(0.1, 1.2, shape=4)'
    },
    'small_conv': {
        'widths': 'loguniform(32, 512, discrete=True, shape=2)',
        'batch_norm': 'choices([True, False])'
    },
    'large_conv': {
        'width': 'loguniform(32, 128, discrete=True)',
        'reductions': 'uniform(0.9, 2.1, shape=4)',
        'batch_norm': 'choices([True, False])'
    }
}


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
        'minidl',
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
        'model': MODEL_SPACES[model],
        'optimizer': OPTIMIZER_SPACES[optimizer]}

    full_space_config['model'].update(space_config.get('model', {}))
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


def get_function(problem_config):
    resources = {
        'cpu': 6, 'mem': '10GB', 'gpu': 1,
        'usage': {'cpu': {'util': 15, 'memory': 2 * 2 ** 30},
                  'gpu': {'util': 10, 'memory': 2 ** 30}}}

    if mahler:
        return mahler.operator(resources=resources)(minidl_run, problem_config=problem_config)
    else:
        return functools.partial(minidl_run, problem_config=problem_config)


if train is not None:
    def build(datasets=None, dataset_folds=None, models=None, optimizers=None, **kwargs):
        return MiniDLBenchmark(datasets, dataset_folds, models, optimizers)
