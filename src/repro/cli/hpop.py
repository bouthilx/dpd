from collections import defaultdict
import argparse
import itertools
import functools
import logging
import os
import pprint
import json
import yaml

from typing import List

try:
    import mahler.client as mahler
except ImportError:
    mahler = None

from repro.benchmark.base import build_benchmark, build_problem, build_benchmark_subparsers

import repro.hpo.trial.base
from repro.hpo.dispatcher.base import build_dispatcher
from repro.hpo.resource.base import build_resource_manager
from repro.hpo.trial.base import build_trial
from repro.hpo.manager import HPOManager
from repro.utils.nesteddict import nesteddict
from repro.utils.checkpoint import resume_from_checkpoint

LOG_FORMAT = '%(asctime)s:%(name)s:%(message)s'
logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = 'configs/hpop'


def main(argv=None):

    # NOTE: When implementing full pipeline, config will become dynamic and change based on which
    # (dataset, model) pair to run
    parser = argparse.ArgumentParser(description='Script to execute benchmarks')

    parser.add_argument(
        '-v', '--verbose',
        action='count', default=0,
        help="logging levels of information about the process (-v: INFO. -vv: DEBUG)")

    subparsers = parser.add_subparsers(dest='command', title='subcommands',
                                       description='subcommands')
    execute_subparser = subparsers.add_parser('execute')
    execute_subparsers = execute_subparser.add_subparsers(
        dest='benchmark', title='benchmark', description='benchmark', help='')

    execute_subparser.add_argument('--save-out', type=str, default=None)
    execute_subparser.add_argument('--checkpoint', type=str, default=None,
                                   help='enable checkpointing & provide a file name')
    execute_subparser.add_argument('--no-resume', action='store_true',
                                   help='do not resume the checkpoint')
    execute_subparsers = build_benchmark_subparsers(execute_subparsers)

    for subparser in execute_subparsers:
        subparser.add_argument(
            '--backend', type=str, default='builtin',
            choices=list(repro.hpo.trial.base.factories.keys()),
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
            '--config-dir-path', type=str, default=DEFAULT_CONFIG_PATH,
            help='Directory with the configuration of the HPO algorithms.')

    for execute_subparser in execute_subparsers:
        continue

    # TODO: Add these arguments only for Mahler backend
    for register_subparser in execute_subparsers:
        register_subparser.add_argument('--version', type=str)
        register_subparser.add_argument('--container', type=str)
        register_subparser.add_argument('--delay', action='store_true')

    #     register_subparser.add_argument(
    #         '--force', action='store_true', default=False,
    #         help='Register even if another similar task already exists')

    for visualize_subparser in visualize_subparsers:
        visualize_subparser.add_argument(
            '--output', type=str,
            default='f{id:03d}-d{dimension:03d}.png')

    options = parser.parse_args(argv)

    if options.backend == 'mahler':
        assert options.container is not None
        assert options.version is not None
    else:
        options.version = '42'

    levels = {0: logging.WARNING,
              1: logging.INFO,
              2: logging.DEBUG}
    logging.basicConfig(level=levels.get(options.verbose, logging.DEBUG), format=LOG_FORMAT)

    benchmark = build_benchmark(name=options.benchmark, **vars(options))

    if options.command == 'execute' and options.delay:
        return delay(benchmark, options)
    elif options.command == 'execute':
        return execute(benchmark, options)
    elif options.command == 'visualize':
        visualize(benchmark, options)
    else:
        raise ValueError('Invalid command: {}'.format(options.command))


def load_config(config_dir_path, benchmark, task, hpo_role, name):
    config_path = os.path.join(config_dir_path, "{benchmark}/{task}/{role}/{name}.yaml").format(
        benchmark=benchmark, task=task, role=hpo_role, name=name)

    if os.path.exists(config_path):
        logger.info(f'Loading config {config_path}')

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        logger.info(f'config file not found {config_path}')
        config = {'name': name}

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        config['max_trials'] = 5

    return config


def create_tags(version, dispatcher, configurator, seed, workers, problem):
    env_tags = (f'd-{dispatcher} c-{configurator} s-{seed} w-{workers}').split(' ')
    return [version] + env_tags + problem.tags


# TODO: Turn this into a generator, on which we can either execute, or register

def iterate(benchmark, options):

    problem_configurations = itertools.product(
        benchmark.problems, options.dispatchers,
        options.configurators, options.seeds, options.workers)

    for problem, dispatcher, configurator, seed, workers in problem_configurations:
        logger.info(f'{dispatcher} - {configurator} - workers:{workers} - seed:{seed} - '
                    f'problem:{problem.tags}')

        dispatcher_config = load_config(options.config_dir_path, benchmark.name, 'hpo',
                                        'dispatcher', dispatcher)
        configurator_config = load_config(options.config_dir_path, benchmark.name, 'hpo',
                                          'configurator', configurator)

        for config in [dispatcher_config, configurator_config]:
            config['seed'] = seed
            config['max_trials'] = options.max_trials

        dispatcher_config['configurator_config'] = configurator_config

        tags = create_tags(options.version, dispatcher, configurator, seed, workers, problem)

        yield dict(dispatcher_config=dispatcher_config,
                   problem=problem,
                   max_trials=options.max_trials,
                   workers=workers, backend=options.backend, container=options.container, tags=tags,
                   checkpoint=options.checkpoint, resume=not options.no_resume)


def checkpoint_key(tags):
    tags = list(tags)
    tags.sort()

    import hashlib
    sh = hashlib.sha256()
    for tag in tags:
        sh.update(tag.encode('utf-8'))

    return sh.hexdigest()[:15]


def execute(benchmark, options):
    logger.info(f'Benchmark: {benchmark.name}')

    optim_data = nesteddict()

    for config in iterate(benchmark, options):
        results = execute_problem(**config)

        # print(results)
        tags = config['tags']
        problem_key = checkpoint_key(tags)
        # results = process_trials(trials)

        optim_data['index'][problem_key] = tags
        optim_data['results'][problem_key] = results

    if not options.delay and options.save_out is not None:
        data = json.dumps(optim_data, indent=4)
        json_file = open(options.save_out, 'w')
        json_file.write(data)
        json_file.write('\n')
        json_file.close()
    elif not options.delay:
        pprint.pprint(optim_data)


def delay(benchmark, options):
    assert options.backend == 'mahler'
    assert mahler is not None

    import repro.cli.hpop

    operator = mahler.operator(resources={
        'cpu': 1, 'mem': '1GB',
        'usage': {'cpu': {'util': 20, 'memory': 2**30}}})
    hpo_operator = operator(repro.cli.hpop.mahler_execute_problem)

    mahler_client = mahler.Client()
    for config in iterate(benchmark, options):
        config['problem_config'] = dict(name=benchmark.name, config=config.pop('problem').config)
        mahler_client.register(hpo_operator.delay(**config), tags=['master'] + config['tags'], container=options.container)


def init_checkpoint(backend, manager, checkpoint_dir, resume, tags):
    if backend == 'builtin':
        init_builtin_checkpoint(manager, checkpoint_dir, resume, tags)
    elif backend == 'mahler':
        init_builtin_checkpoint(manager, checkpoint_dir, resume, tags)
    else:
        raise NotImplementedError


def init_builtin_checkpoint(manager, checkpoint_dir, resume, tags):

    if not checkpoint_dir:
        return

    problem_hex = checkpoint_key(tags)
    chk_file = os.path.join(checkpoint_dir, problem_hex)

    logger.info(f'Looking for previous checkpoint at {chk_file}: ...')

    if os.path.exists(chk_file):
        logger.info('Checkpoint was found')

        if resume:
            logger.info('Resuming ...')
            manager = resume_from_checkpoint(manager, chk_file)
        else:
            logger.warning('Ignoring Checkpoint file.. It will be overridden!')
    else:
        logger.info('No Checkpoint!')

    if checkpoint_dir:
        file_name = chk_file

        logger.info(f'Enabling checkpoints in {file_name}')

        manager.enable_checkpoints(
            name=problem_hex,
            every=None,
            archive_folder=checkpoint_dir
        )


# def init_mahler_checkpoint(manager, checkpoint_dir, tags):
#     # Load all trials, reobserve, tada.
#     mahler_client = mahler.Client()
#     task_id = mahler.get_current_task_id()
#     # TODO: Add projection to limit I/O
#     for task_doc in mahler_client.find(tags=['worker', task_id]):
#         hpo_trial = manager.trial_factory(task_doc['name'], manager.task, manager.manager.Queue())
#         # use manager.trial_factory?
#         manager.suspended_trials.add(hpo_trial)
#         hpo.start()  # initiate metric transfert
#         manager.dispatcher.observe(hpo_trial, trial.results)
#
#     manager.dispatcher.trial_count = len(trials)


def mahler_execute_problem(dispatcher_config, problem_config, max_trials, workers, backend,
                           container, tags, checkpoint, resume):

    results = execute_problem(
        dispatcher_config, build_problem(**problem_config), max_trials, workers, backend,
        container, tags, checkpoint, resume)

    return dict(results=results)


def execute_problem(dispatcher_config, problem, max_trials, workers, backend, container,
                    tags, checkpoint, resume):

    backend_config = dict(tags=tags, container=container)

    dispatcher = build_dispatcher(problem.space, **dispatcher_config)

    resource_manager = build_resource_manager(backend, workers=workers, operator=problem.run,
                                              **backend_config)
    trial_factory = functools.partial(build_trial, name=backend, **backend_config)

    manager = HPOManager(resource_manager, dispatcher, problem.run, trial_factory,
                         max_trials=max_trials, workers=workers)

    init_checkpoint(backend, manager, checkpoint, resume, tags)

    manager.run()
    resource_manager.terminate()

    return [trial.to_dict() for trial in manager.trials]


def plot(trials, observations):

    import matplotlib.pyplot as plt

    fig = plt.figure()
    axes = fig.subplots(nrows=3, ncols=1)  # , sharex=True)

    density = defaultdict(int)

    hpo_objectives = []
    n = 0

    for trial in sorted(trials, key=lambda trial: trial.creation_time):
        n += 1
        trial_observations = observations[trial.id]

        x = list(sorted(filter(lambda step: step > 0, trial_observations.keys())))
        y = [trial_observations[i] for i in x]

        if not y:
            continue

        for i in x:
            density[i] += 1

        min_y = min(y)
        if hpo_objectives:
            hpo_objectives.append(min(hpo_objectives[-1], min_y))
        else:
            hpo_objectives.append(min_y)

        axes[1].plot(x, y, color='blue')

    axes[0].plot(range(len(hpo_objectives)), hpo_objectives)
    # axes[0].set_yscale('log')

    x, y = list(zip(*list(sorted(density.items()))))
    axes[2].plot(x, [int(v / n * 100 + 0.5) for v in y])

    plt.show()


def visualize(benchmark, options):
    raise NotImplementedError


if __name__ == "__main__":
    main()
