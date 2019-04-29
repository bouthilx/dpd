from collections import defaultdict
import argparse
import copy
import itertools
import logging
import math
import os
import pprint
import re
import json

import yaml

try:
    import mahler.client as mahler
except ImportError:
    mahler = None

from repro.benchmark.base import build_benchmark, build_benchmark_subparsers
import repro.hpo.configurator.base
import repro.hpo.dispatcher.base
from repro.hpo.dispatcher.dispatcher import HPOManager
from repro.utils.nesteddict import nesteddict


logger = logging.getLogger(__name__)


def main(argv=None):

    # NOTE: When implementing full pipeline, config will become dynamic and change based on which
    # (dataset, model) pair to run
    parser = argparse.ArgumentParser(description='Script to execute benchmarks')

    parser.add_argument(
        '-v', '--verbose',
        action='count', default=0,
        help="logging levels of information about the process (-v: INFO. -vv: DEBUG)")

    subparsers = parser.add_subparsers(dest='command', title='subcommands', description='subcommands', help='')
    execute_subparser = subparsers.add_parser('execute')
    execute_subparsers = execute_subparser.add_subparsers(
        dest='benchmark', title='benchmark', description='benchmark', help='')

    execute_subparser.add_argument('--json-file', type=str, default=None)
    execute_subparsers = build_benchmark_subparsers(execute_subparsers)
    
    for subparser in execute_subparsers:
        subparser.add_argument(
            '--backend', type=str, default='builtin', choices=['builtin', 'mahler'],
            help=('Backend to run jobs in parallel. Large benchmarks may not scale well with '
                  'builtin backend.'))

    visualize_subparser = subparsers.add_parser('visualize')
    visualize_subparsers = visualize_subparser.add_subparsers(
        dest='benchmark', title='benchmark', description='benchmark', help='')
    visualize_subparsers = build_benchmark_subparsers(visualize_subparsers)

    for subparser in itertools.chain(execute_subparsers, visualize_subparsers):
        subparser.add_argument(
            '--dispatchers', type=str, required=True, nargs='*',
            choices=list(repro.hpo.dispatcher.base.factories.keys()))
        subparser.add_argument(
            '--configurators', type=str, required=True, nargs='*',
            choices=list(repro.hpo.configurator.base.factories.keys()))
        subparser.add_argument(
            '--workers', type=int, nargs='*', default=[1])
        subparser.add_argument(
            '--max-trials', type=int, default=100)
        subparser.add_argument(
            '--seeds', type=int, nargs='*', default=[1])
        # subparser.add_argument(
        #     '--seed', type=int, default=1,
        #     help='Seed for the benchmark.')

    for subparser in execute_subparsers:
        subparser.add_argument(
            '--config-dir-path', required=True,
            help='Directory with the configuration of the HPO algorithms.')

    for execute_subparser in execute_subparsers:
        continue

    # TODO: Add these arguments only for Mahler backend
    # for register_subparser in register_subparsers:
    #     register_subparser.add_argument('--version', type=str, required=True)
    #     register_subparser.add_argument('--container', type=str, required=True)
    #     register_subparser.add_argument(
    #         '--force', action='store_true', default=False,
    #         help='Register even if another similar task already exists')

    for visualize_subparser in visualize_subparsers:
        visualize_subparser.add_argument('--version', type=str, required=True)
        visualize_subparser.add_argument(
            '--output', type=str,
            default='f{id:03d}-d{dimension:03d}.png')

    options = parser.parse_args(argv)

    levels = {0: logging.WARNING,
              1: logging.INFO,
              2: logging.DEBUG}
    logging.basicConfig(level=levels.get(options.verbose, logging.DEBUG))

    benchmark = build_benchmark(name=options.benchmark, **vars(options))

    if options.command == 'execute':
        execute(benchmark, options)
    elif options.command == 'visualize':
        visualize(benchmark, options)
    else:
        raise ValueError('Invalid command: {}'.format(options.command))


def load_config(config_dir_path, benchmark, hpo_role, name):
    config_path = os.path.join(config_dir_path, "{benchmark}/{role}/{name}.yaml").format(
        benchmark=benchmark, role=hpo_role, name=name)

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {'name': name}

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        config['max_trials'] = 5

    return config



def execute(benchmark, options):
    print(f'Benchmark: {benchmark.name}')

    optim_data = nesteddict()
    problem_configurations = itertools.product(benchmark.problems, options.dispatchers,
                                               options.configurators, options.seeds, options.workers)

    for problem, dispatcher, configurator, seed, workers in problem_configurations:
        print(f'{dispatcher} - {configurator} - workers:{workers} - seed:{seed} - problem:{problem.tags}')

        dispatcher_config = load_config(options.config_dir_path, benchmark.name, 'dispatcher',
                                        dispatcher)
        configurator_config = load_config(options.config_dir_path, benchmark.name, 'configurator',
                                          configurator)

        for config in [dispatcher_config, configurator_config]:
            config['seed'] = seed
            config['max_trials'] = options.max_trials

        dispatcher_config['configurator_config'] = configurator_config

        trials = execute_problem(dispatcher_config, problem, options.max_trials, workers)

        results = process_trials(trials)

        optim_data[','.join(problem.tags)][dispatcher][configurator][seed][workers] = results

    if options.json_file is not None:
        data = json.dumps(optim_data, indent=4)
        json_file = open(options.json_file, 'w')
        json_file.write(data)
        json_file.write('\n')
        json_file.close()
    else:
        pprint.pprint(optim_data)


def process_trials(trials):
    results = []
    # TODO: Sort by creation time
    for trial in trials:
        results.append({'params': trial.params, 'objective': trial.get_last_results()[-1]['objective']})

    return results


def execute_problem(dispatcher_config, problem, max_trials, workers):

    dispatcher = repro.hpo.dispatcher.base.build_dispatcher(problem.space, **dispatcher_config)

    manager = HPOManager(dispatcher, problem.run, max_trials=max_trials, workers=workers)
    manager.run()

    return manager.trials


def visualize(benchmark, options):
    raise NotImplementedError


if __name__ == "__main__":
    main()
