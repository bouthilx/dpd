import copy
import itertools
import random

from orion.core.io.space_builder import Space, DimensionBuilder

import numpy

from repro.utils.flatten import flatten, unflatten

try:
    from repro.train import train
except ImportError:
    train = None

try:
    import mahler.client as mahler
    from mahler.core.utils.errors import SignalInterruptTask, SignalSuspend
except ImportError:
    mahler = None


class MiniDLBenchmark:

    name = 'minidl'

    def __init__(self, datasets=None, dataset_folds=None, models=None, optimizers=None,
                 scenarios=None, previous_tags=None):
        self.verify(datasets, dataset_folds, models, optimizers, scenarios)
        self._datasets = datasets
        self._dataset_folds = dataset_folds
        self._models = models
        self._optimizers = optimizers
        self._scenarios = scenarios
        self.previous_tags = previous_tags

    def verify(self, datasets, dataset_folds, models, optimizers, scenarios):
        if datasets is not None:
            valid_datasets = set(self.datasets)
            for dataset in datasets:
                assert dataset in valid_datasets, dataset

        if dataset_folds is not None:
            valid_dataset_folds = set(self.dataset_folds)
            for dataset_fold in dataset_folds:
                assert dataset_fold in valid_dataset_folds, dataset_fold

        if models is not None:
            valid_models = set(self.models)
            for model in models:
                assert model in valid_models, model

        if optimizers is not None:
            valid_optimizers = set(self.optimizers)
            for optimizer in optimizers:
                assert optimizer in valid_optimizers, optimizer

        if scenarios is not None:
            valid_scenarios = set(self.scenarios)
            for scenario in scenarios:
                assert scenario in valid_scenarios, scenario

    def add_subparser(self, subparsers):

        benchmark_parser = subparsers.add_parser('minidl')

        benchmark_parser.add_argument('--datasets', choices=self.datasets, nargs='*', type=str)
        benchmark_parser.add_argument('--dataset-folds', choices=self.dataset_folds, nargs='*',
                                      type=int)
        benchmark_parser.add_argument('--models', choices=self.models, nargs='*', type=str)
        benchmark_parser.add_argument('--optimizers', choices=self.optimizers, nargs='*',
                                      type=str)
        benchmark_parser.add_argument('--scenarios', choices=self.scenarios, nargs='*', type=str)

        return benchmark_parser

    @property
    def datasets(self):
        if getattr(self, '_datasets', None):
            return self._datasets

        return ['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'tinyimagenet']

    @property
    def dataset_folds(self):
        if getattr(self, '_dataset_folds', None):
            return self._dataset_folds

        return list(range(1, 6))

    @property
    def models(self):
        if getattr(self, '_models', None):
            return self._models

        return ['logreg', 'small_mlp', 'large_mlp', 'small_conv', 'large_conv']

    @property
    def optimizers(self):
        if getattr(self, '_optimizers', None):
            return self._optimizers

        return ['sgd', 'adam']

    @property
    def scenarios(self):
        if getattr(self, '_scenarios', None):
            return self._scenarios

        return ['0.0',  # No diff
                '1.1',  # Diff dataset
                '1.2',  # Diff model
                '1.3',  # Diff optimizer
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
                '3.5',  # H-P is scaled
                '3.6']  # Bug causing leak of valid set

    @property
    def problems(self):
        problems_iterator = itertools.product(self.datasets, self.dataset_folds, self.models,
                                              self.optimizers, self.scenarios)
        return [Problem(*(args + (self.previous_tags, ))) for args in problems_iterator]



# TODO: Seeding of configurator!!

# We should be able to change the model hyper-parameters space definition
# Also for the optimizer hyper-parameter space definition
# We should be able to specify if loss-scale, loss-invertion or hyper-parameter scale or invertion
#
# Model H-P, width, dropout
# Optimizer H-P, lr, momentum, beta, weight-decay
#
# For convenience, seed of (model, data-order) is equal to number of samples (counting history)
# This way all algorithms see the same models, but they don't optimize for a specific seed.
# For multiple runs, we will use this
# Take range(500), shuffle randomly. Take [:100] for run 1, [100:200] for run 2, etc.

# Nah, pass seed to init. Then sample seeds to exps.
# For simplicity, seed = nb of runs. This makes it similar to coco benchmark where when cannot
# seed, but use 1-5 instances for _randomization_.

# How to make h-p optimization reproducible?
# We need some form of synchronization.
# We could avoid creating new jobs until all pool is done.
# Totally avoid parallelism for simplicity?
#


class Problem:
    def __init__(self, dataset_name, dataset_fold, model_name, optimizer_name, scenario,
                 previous_tags):
        self.dataset_name = dataset_name
        self.dataset_fold = dataset_fold
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.scenario = scenario
        self.previous_tags = previous_tags

        # TODO: make it sequential for all problems...
        self.model_seed = 1
        self.sampler_seed = 1

    @property
    def tags(self):
        tags = ['minidl',
                'd-{}'.format(self.dataset_name),
                'df-{}'.format(self.dataset_fold),
                'm-{}'.format(self.model_name),
                'o-{}'.format(self.optimizer_name),
                's-{}'.format(self.scenario)]
        if self.previous_tags:
            tags += ['pv-{}'.format(tag) for tag in self.previous_tags]
        return tags

    @property
    def config(self):
        return {
            'data': {
                'name': self.dataset_name,
                'fold': self.dataset_fold,
                'mini': True,
                'batch_size': 128},
            'model': {
                'name': self.model_name},
            'optimizer': OPTIMIZER_CONFIGS[self.optimizer_name],
            'model_seed': self.model_seed,
            'sampler_seed': self.sampler_seed
            }

    def execute(self, configurator_config):
        print('EXECUTING!')
        print(self.config)
        scenario_update = SCENARIO_UPDATES.get(self.scenario, {})
        if scenario_update:
            space_config = {
                'model': scenario_update.get('model', {}).get(self.model_name, {}),
                'optimizer': scenario_update.get('optimizer', {}).get(self.optimizer_name, {})}
        else:
            space_config = {}

        execute(self.config, space_config, configurator_config, self.previous_tags)

    def register(self, mahler_client, configurator_config, container, tags):
        if mahler is None:
            print('`mahler` is not installed. Cannot register.')

        # TODO: Add support for previous_tags
        mahler_client.register(
            create_trial.delay(
                problem_config=self.config,
                space_config={},
                configurator_config=configurator_config),
            container=container, tags=tags)

        print('Registered', *tags)


if train is not None:
    def build(datasets=None, dataset_folds=None, models=None, optimizers=None, scenarios=None, **kwargs):
        return MiniDLBenchmark(datasets, dataset_folds, models, optimizers, scenarios)


# Scenarios
# 0. No diff
# 1. Diff environment
#     1.1 Dataset
#     1.2 Model
#     1.3 Optimizer
# (requires change of config)
# 2. H-Ps
#     2.1 fewer H-Ps
#     2.2 More H-Ps
#     2.3 Prior changed with half overlap
#     2.4 Prior changed with no overlap
# (requires flag to change)
# 3. Code change
#     3.1 Without any effect
#     3.2 Loss is reversed
#     3.3 Loss is scaled
#     3.4 HP is reversed
#     3.5 HP is scaled
#     3.6 Bug causing leak of valid set


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
        'weight_decay': 'loguniform(1.0e-10, 1.0e-3)'
    },
    'adam': {
        'lr': 'loguniform(1.0e-5, 1.0)',
        'betas': 'loguniform(0.7, 1.0, shape=2)',
        'weight_decay': 'loguniform(1.0e-10, 1.0e-3)'
    }
}


#####
# Scenarios
#####

SCENARIO_UPDATES = {
    # Remove weight decay
    "2.1": {
        'optimizer': {
            'sgd': {'weight_decay': None}}},
    # Add momentum
    "2.2": {
        'optimizer': {
            'sgd': {'momentum': 'uniform(0, 1.0)'}}},
    # Change prior
    "2.3.a": {
        'optimizer': {
            'sgd': {'lr': 'uniform(0.0001, 0.1)'}}},
    "2.3.b": {
        'optimizer': {
            'sgd': {'lr': 'loguniform(0.0001, 0.1)'}}}}


def build_space(config, **space_config):
    config['model']['name']
    config['optimizer']['name']

    space = Space()

    dimension_builder = DimensionBuilder()

    full_space_config = {
        'model': MODEL_SPACES[config['model']['name']],
        'optimizer': OPTIMIZER_SPACES[config['optimizer']['name']]}

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


def cure_narray(config):
    cured_config = {}
    for key, value in flatten(config).items():
        if isinstance(value, numpy.ndarray):
            value = list(value)

        cured_config[key] = value

    return unflatten(cured_config)


def execute(problem_config, space_config, configurator_config, previous_tags=None):

    space = build_space(problem_config, **space_config)
    configurator = build_hpo(space, **configurator_config)

    if previous_tags is not None:
        # TODO: Load previous tags, make the configurator observe them

        # And then we increment max_trials otherwise the configurator would already return
        # is_completed() -> True
        configurator.max_trials += len(configurator.trials)

    numpy.random.seed(problem_config['data']['fold'])
    seeds = numpy.random.randint(1, 1000000, size=configurator_config['max_trials'])

    problem_config['max_epochs'] = 1

    objectives = []

    while not configurator.is_completed():
        seed = seeds[len(configurator.trials)]
        random.seed(seed)
        params = configurator.get_params(seed=seed)
        params['model_seed'] = params['sampler_seed'] = seed
        rval = train(**merge(problem_config, params))
        objective = rval['best']['valid']['error_rate']
        configurator.observe([dict(params=params, objective=objective)])
        objectives.append(objective)

    return dict(objectives=objectives)


# register create trial
# create-trial trains configurator on completed trials up to n - pool_size
# to have a sequential-like generation of trials
#


def create_trial(problem_config, space_config, configurator_config, instance=1, previous_tags=None):
    space = build_space(problem_config, **space_config)
    configurator = build_hpo(space, **configurator_config)

    if previous_tags is not None:
        # TODO: Load previous tags, make the configurator observe them

        # And then we increment max_trials otherwise the configurator would already return
        # is_completed() -> True
        configurator.max_trials += len(configurator.trials)

    numpy.random.seed(instance)
    seeds = numpy.random.randint(1, 1000000, size=configurator_config['max_trials'])

    # TODO: Remove after debug
    problem_config['max_epochs'] = 1

    # Create client inside function otherwise MongoDB does not play nicely with multiprocessing
    mahler_client = mahler.Client()

    task = mahler_client.get_current_task()
    tags = [tag for tag in task.tags if tag != task.name]
    container = task.container

    n_broken = 0
    projection = {'output': 1, 'arguments': 1, 'registry.status': 1}
    trials = mahler_client.find(tags=tags + [train.name, 'hpo'], _return_doc=True,
                                _projection=projection)

    n_uncompleted = 0
    n_trials = 0

    # NOTE: Only support sequential for now. Much simpler.
    objectives = []
    print('---')
    print("Training configurator")
    for trial in trials:
        n_trials += 1
        trial = convert_mahler_task_to_trial(trial)
        if trial['status'] == 'Cancelled':
            continue

        completed_but_broken = trial['status'] == 'Completed' and not trial['objective']
        # Broken
        if trial['status'] == 'Broken' or completed_but_broken:
            n_broken += 1
        # Uncompleted
        elif trial['status'] != 'Completed':
            n_uncompleted += 1
        # Completed
        else:
            configurator.observe([trial])
            objectives.append(trial['objective'])

    print('---')
    print('There is {} trials'.format(n_trials))
    print('{} uncompleted'.format(n_uncompleted))
    print('{} completed'.format(len(configurator.trials)))
    print('{} broken'.format(n_broken))

    if n_broken > 10:
        message = (
            '{} trials are broken. Suspending creation of trials until investigation '
            'is done.').format(n_broken)

        mahler_client.close()
        raise SignalSuspend(message)

    if configurator.is_completed():
        mahler_client.close()
        return dict(objectives=objectives)

    print('---')
    print("Generating new configurations")
    if n_uncompleted == 0:
        # NOTE: Only completed trials are in configurator.trials
        seed = int(seeds[len(configurator.trials)])
        random.seed(seed)
        params = configurator.get_params(seed=seed)
        params['model_seed'] = params['sampler_seed'] = seed
        trial_config = cure_narray(merge(problem_config, params))
        mahler_client.register(train.delay(**trial_config), container=container,
                               tags=tags + ['hpo'])
    else:
        print('A trial is pending, waiting for its completion before creating a new trial.')

    mahler_client.close()
    raise SignalInterruptTask('HPO not completed')


def convert_mahler_task_to_trial(trial):
    output = trial.get('output', {})
    if output:
        objective = output.get('best', {}).get('valid', {}).get('error_rate', None)
    else:
        objective = None

    return dict(id=trial['id'], status=trial['registry']['status'], params=trial['arguments'],
                objective=objective)


if mahler is not None and train:
    create_trial = mahler.operator(resources={'cpu': 2, 'mem': '20MB'},
                                   resumable=True)(create_trial)
    train = mahler.operator(resources={'cpu': 4, 'gpu': 1, 'mem': '20BG'}, resumable=True)(train)
