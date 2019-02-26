"""
coco benchmark -> one task per (problem, dimensions, instance)

mini-dl -> one task generator per (dataset, model)

full-dl -> one task generator per (dataset, model)
"""
import argparse
import copy
import itertools
import logging
import math
import os

import yaml

try:
    import mahler.client as mahler
except ImportError:
    mahler = None

from repro.benchmark.base import build_benchmark, build_benchmark_subparsers
import repro.hpo.base


logger = logging.getLogger(__name__)


# NOTE: Default path when running in the container
DEFAULT_CONFIG_DIR_PATH = '/repos/repro/configs/repro/zoo/'

DATASET_NAMES = ['mnist', 'fashionmnist', 'svhn', 'cifar10', 'cifar100', 'emnist', 'tinyimagenet']

MODEL_NAMES = (
    ['lenet', 'mobilenet', 'mobilenetv2'] +
    ['vgg{}'.format(i) for i in [11, 13, 16, 19]] +
    ['densenet{}'.format(i) for i in [121, 161, 169, 201]] +
    ['resnet{}'.format(i) for i in [18, 34, 50, 101]] +
    ['preactresnet{}'.format(i) for i in [18, 34, 50, 101]])


def total_trials(max_epochs, reduction_factor, max_resource, fidelity_space, **kwargs):
    fidelity_levels = next(iter(fidelity_space.values()))
    total_epochs = 0
    for i, n_epochs in zip(range(len(fidelity_levels) - 1, -1, -1), fidelity_levels):
        total_epochs += n_epochs * max_resource * reduction_factor ** i

    return int(math.ceil(total_epochs / max_epochs))


MAX_EPOCHS = 120
MAX_WORKERS = 10
MAX_RESOURCE = 20
NUMBER_OF_SEEDS = 20

asha_config = dict(name='asha', reduction_factor=4, max_resource=40,
                   fidelity_space=dict(max_epochs=[15, 30, 60]))

# asha_config = dict(name='asha', reduction_factor=1, max_resource=MAX_RESOURCE,
#                    fidelity_space=dict(max_epochs=[1, 2, 4]))

random_search_config = dict(name='random_search',
                            max_trials=total_trials(MAX_EPOCHS, **asha_config))

configurator_configs = dict(asha=asha_config, random_search=random_search_config)



def main(argv=None):

    # NOTE: When implementing full pipeline, config will become dynamic and change based on which
    # (dataset, model) pair to run
    parser = argparse.ArgumentParser(description='Script to execute or deploy benchmarks')

    parser.add_argument(
        '-v', '--verbose',
        action='count', default=0,
        help="logging levels of information about the process (-v: INFO. -vv: DEBUG)")

    subparsers = parser.add_subparsers(dest='command', title='subcommands', description='subcommands', help='')
    execute_subparser = subparsers.add_parser('execute')
    execute_subparsers = execute_subparser.add_subparsers(
        dest='benchmark', title='benchmark', description='benchmark', help='')
    execute_subparsers = build_benchmark_subparsers(execute_subparsers)

    if mahler is not None:
        register_subparser = subparsers.add_parser('register')
        register_subparsers = register_subparser.add_subparsers(
            dest='benchmark', title='benchmark', description='benchmark', help='')
        register_subparsers = build_benchmark_subparsers(register_subparsers)
    else:
        print('Mahler is not installed, cannot register benchmarks.')
        register_subparsers = []

    for subparser in itertools.chain(execute_subparsers, register_subparsers):
        subparser.add_argument(
            '--configurators', type=str, required=True, nargs='*',
            choices=list(repro.hpo.base.factories.keys()))
        # subparser.add_argument(
        #     '--seed', type=int, default=1,
        #     help='Seed for the benchmark.')
        subparser.add_argument(
            '--config-dir-path', required=True,
            help='Directory with the configuration of the HPO algorithms.')

    for execute_subparser in execute_subparsers:
        continue

    for register_subparser in register_subparsers:
        register_subparser.add_argument('--version', type=str, required=True)
        register_subparser.add_argument('--container', type=str, required=True)
        register_subparser.add_argument(
            '--force', action='store_true', default=False,
            help='Register even if another similar task already exists')

    options = parser.parse_args(argv)

    levels = {0: logging.WARNING,
              1: logging.INFO,
              2: logging.DEBUG}
    logging.basicConfig(level=levels.get(options.verbose, logging.DEBUG))

    benchmark = build_benchmark(name=options.benchmark, **vars(options))

    if options.command == 'register':
        register(benchmark, options)
    elif options.command == 'execute':
        execute(benchmark, options)
    else:
        raise ValueError('Invalid command: {}'.format(options.command))


def load_config(config_dir_path, benchmark, configurator):
    config_path = os.path.join(config_dir_path, "{benchmark}/{configurator}.yaml").format(
        benchmark=benchmark, configurator=configurator)

    with open(config_path, 'r') as f:
        config = yaml.load(f)

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        config['max_trials'] = 5

    return config


def is_registered(mahler_client, problem, supp_tags):
    return any(True for _ in mahler_client.find(tags=problem.tags + supp_tags))


def register(benchmark, options):

    mahler_client = mahler.Client()

    logger.debug(benchmark.problem_ids)

    for problem, configurator in itertools.product(benchmark.problems, options.configurators):
        configurator_config = load_config(options.config_dir_path, benchmark.name, configurator)
        logger.debug(problem.tags)

        supp_tags = [configurator, options.version, 'hpo-transfer']

        if is_registered(mahler_client, problem, supp_tags) and not options.force:
            logger.warning('HPO `{}` already registered for tags: {}'.format(
                configurator, ", ".join(problem.tags + supp_tags)))
            continue

        # Register problem used to warm-start
        previous_problem = copy.deepcopy(problem)
        previous_problem.scenario = SCENARIOS[problem.scenario][0]
        previous_problem.previous_tags = SCENARIOS[problem.scenario][1]

        if not is_registered(mahler_client, previous_problem, supp_tags):
            previous_problem.register(mahler_client, configurator_config, options.container,
                                      previous_problem.tags + supp_tags)

        # Register an equivalent problem which won't be warm-started
        # In 0.0 there is no changes, so the previous problem would be the same as the cold turkey,
        # hence we don't need the cold turkey.
        if problem.scenario != "0.0":
            cold_turkey = copy.deepcopy(problem)
            cold_turkey.previous_tags = None

            if not is_registered(mahler_client, cold_turkey, supp_tags):
                previous_problem.register(mahler_client, configurator_config, options.container,
                                          cold_turkey.tags + supp_tags)

        # Make deep copy because same problem is bundled with different cofigurators
        problem = copy.deepcopy(problem)
        # Don't want pv-hpo-transfer
        problem.previous_tags = previous_problem.tags + supp_tags[:-1]

        # Register the problem that will be warm-started based on results of `previous_problem`
        problem.register(mahler_client, configurator_config, options.container,
                         problem.tags + supp_tags)


def execute(benchmark, options):
    print(benchmark.scenarios)
    for problem, configurator in itertools.product(benchmark.problems, options.configurators):
        print(problem.scenario)
        configurator_config = load_config(options.config_dir_path, benchmark.name, configurator)

        problem.execute(configurator_config)


SCENARIOS = {
    "0.0": ["0.0", None],
    "2.3.a": ["0.0", None],
    "2.3.b": ["0.0", None],
    "2.4.a": ["2.4.b", None],
    "2.4.b": ["2.4.a", None]}



# if scenario 0.
#     if s-0 pv-none v does not exists: register.
#     register s-0 pv-v
# if scenario 1. ?
# if scenario 2.1 ?
# if scenario 2.2 ?
# if scenario 2.3 ?
# if scenario 2.4.a
#     if s-2.4.b pv-none v does not exists: register.
#     register s-2.4.a v pv-v pv-s-2.4.b ...
# if scenario 2.4.b
#     if s-2.4.a pv-none v does not exists: register.
#     register s-2.4.b v pv-v pv-s-2.4.a ...








if __name__ == "__main__":
    main()
