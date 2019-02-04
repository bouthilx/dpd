from collections import defaultdict
import copy
import logging
import os
import pprint
import random
import time

import mahler.client as mahler
from mahler.core.utils.flatten import flatten, unflatten
from mahler.core.utils.errors import SignalInterruptTask, SignalSuspend

from orion.core.io.space_builder import Space, DimensionBuilder

import yaml

from repro.train import train
from repro.hpo.base import build_hpo


logger = logging.getLogger(__name__)


run = mahler.operator(resources={'cpu': 4, 'gpu': 1, 'mem': '20BG'}, resumable=True)(train)


# run needs
# data, model, optimizer, model_seed, sampler_seed, max_epochs

# This is build from config_file + args of create_trila


def load_config(config_dir_path, dataset_name, model_name):

    config_path = os.path.join(config_dir_path, "{dataset}/{model}.yaml").format(
        dataset=dataset_name, model=model_name)

    with open(config_path, 'r') as f:
        config = yaml.load(f)

    return config


def clean_duplicates(mahler_client, configurator, new_task, tags):

    if configurator.__class__.__name__.lower() not in ['asha']:
        return

    # NOTE: For now only ASHA may produce duplicates on race conditions.
    asha = configurator

    new_arguments = new_task.arguments
    new_id = new_task.id

    number_of_duplicates = 0
    number_of_duplicates = defaultdict(int)
    originals = dict()
    for task_document in mahler_client.find(tags=tags + ['asha', run.name], _return_doc=True,
                                            _projection={'arguments': 1, 'registry.status': 1}):
        task_arguments = task_document['arguments']
        task_id = task_document['id']

        if task_arguments[asha.fidelity_dim] != new_arguments[asha.fidelity_dim]:
            continue

        if asha._fetch_trial_params(task_arguments) == asha._fetch_trial_params(new_arguments):
            number_of_duplicates[new_id] += 1

            if (number_of_duplicates[new_id] > 1 and
                    task_document['registry']['status'] != 'Cancelled'):
                try:
                    message = 'Duplicate of task {}'.format(originals[new_id]['id'])
                    mahler_client.cancel(task_id, message)
                except Exception:
                    message = "Could not cancel task {}, a duplicate of {}".format(
                        task_id, originals[new_id]['id'])
                    logger.error(message)
            else:
                originals[new_id] = task_document


def compute_args(trials, space):
    min_objective = float('inf')
    max_objective = -float('inf')
    min_args = None
    max_args = None
    mean_args = dict()

    completed_trials = [trial for trial in trials if trial['registry']['status'] == 'Completed']
    for trial in completed_trials:

        try:
            objective = trial['output']['best']['valid']['error_rate']
        except Exception:
            pprint.pprint(trial['output'])
            raise

        if min_objective > objective:
            min_objective = objective
            min_args = trial['arguments']

        if max_objective < objective:
            max_objective = objective
            max_args = trial['arguments']

        flattened_arguments = flatten(trial['arguments'])
        for key in space.keys():
            item = flattened_arguments[key]
            mean_args[key] = item + mean_args.get(key, 0)

    n_completed_trials = len(completed_trials)

    for key, item in mean_args.items():
        mean_args[key] = item / n_completed_trials

    return min_args, max_args, mean_args


def register_best_trials(mahler_client, configurator, tags, container, max_epochs, max_resource,
                         number_of_seeds):

    best_trials = configurator.get_bests(max_resource)
    print('\nBest trials:')
    for trial in sorted(best_trials, key=lambda trial: trial['id']):
        print('{}: {}'.format(trial['id'], trial['registry']['status']))

    if any(trial['registry']['status'] != 'Completed' for trial in best_trials):
        mahler_client.close()
        # Force re-execution of the task until all trials are done
        raise SignalInterruptTask('Not all trials are completed. Rerun the task.')

    projection = {'tags': 1}

    trials = mahler_client.find(tags=tags + ['distrib'], _return_doc=True,
                                _projection=projection)

    existing_trials = dict(min=0, max=0, mean=0)
    for trial in trials:
        for name in ['min', 'max', 'mean']:
            if name in trial['tags']:
                existing_trials[name] += 1
                break

    min_args, max_args, mean_args = compute_args(best_trials, configurator.space)

    # Use first trial, we don't care since they all have the same arguments
    # except for the optimizer HPs.
    config = best_trials[0]['arguments']

    min_config = merge(config, min_args)
    max_config = merge(config, max_args)
    mean_config = merge(config, mean_args)

    print('\nNew trials registered for distribution evaluation:')
    for name, config in [('min', min_config), ('max', max_config), ('mean', mean_config)]:
        for i in range(number_of_seeds - existing_trials[name]):
            # This time we want to test error as well.
            config['compute_test_error_rates'] = True
            config['max_epochs'] = max_epochs  # Just to make sure...

            register_new_trial(mahler_client, config, tags + ['distrib', name], container)


def merge(config, subconfig):
    flattened_config = copy.deepcopy(flatten(config))
    flattened_config.update(flatten(subconfig))
    return unflatten(flattened_config)


def register_new_trial(mahler_client, trial_config, tags, container):
    trial_config = copy.deepcopy(trial_config)
    trial_config['model_seed'] = random.uniform(1, 10000)
    trial_config['sampler_seed'] = random.uniform(1, 10000)
    new_task = mahler_client.register(run.delay(**trial_config), container=container, tags=tags)
    print(new_task.id, sorted(new_task.tags))
    return new_task


def sample_new_config(configurator, config):
    params = configurator.get_params()
    params['optimizer']['lr_scheduler'] = dict(patience=10)
    # Params can be all
    return merge(config, params)


# TODO: Support resources properly
#       (should submit based on largest request and enable running small tasks in large resource
#        workers)
# @mahler.operator(resources={'cpu':1, 'mem':'1GB'})
@mahler.operator(resources={'cpu': 4, 'gpu': 1, 'mem': '20GB'})
def create_trial(config_dir_path, dataset_name, model_name, configurator_config,
                 max_epochs, max_resource, number_of_seeds):

    # Create client inside function otherwise MongoDB does not play nicely with multiprocessing
    mahler_client = mahler.Client()

    task = mahler_client.get_current_task()
    tags = [tag for tag in task.tags if tag != task.name]
    container = task.container

    projection = {'output': 1, 'arguments': 1, 'registry.status': 1}
    # *IMPORTANT* NOTE:
    #     Space must be instantiated in the function otherwise its internal RNG state gets copied
    #     every time a task is forked in a process and each of them will start with the exact
    #     same state, leading to identical sampling. What a naughty side effect!
    space = Space()
    space['optimizer.lr'] = (
        DimensionBuilder().build('optimizer.lr', 'loguniform(1e-5, 0.5)'))
    space['optimizer.momentum'] = (
        DimensionBuilder().build('optimizer.momentum', 'uniform(0., 0.9)'))
    space['optimizer.weight_decay'] = (
        DimensionBuilder().build('optimizer.weight_decay', 'loguniform(1e-8, 1e-3)'))

    config = load_config(config_dir_path, dataset_name, model_name)

    configurator = build_hpo(space, **configurator_config)

    n_broken = 0
    trials = mahler_client.find(tags=tags + [run.name, 'hpo'], _return_doc=True,
                                _projection=projection)

    for trial in trials:
        if trial['registry']['status'] == 'Cancelled':
            continue

        configurator.observe([trial])

        n_broken += int(trial['registry']['status'] == 'Completed' and not trial['output'])
        n_broken += int(trial['registry']['status'] == 'Broken')

    if n_broken > 10:
        message = (
            '{} trials are broken. Suspending creation of trials until investigation '
            'is done.').format(n_broken)

        raise SignalSuspend(message)

    if configurator.is_completed():
        new_best_trials = register_best_trials(
            mahler_client, configurator, tags, container, max_epochs, max_resource, number_of_seeds)
        mahler_client.close()
        return new_best_trials

    config['max_epochs'] = max_epochs
    new_task_config = sample_new_config(configurator, config)
    trial_task = register_new_trial(
        mahler_client, new_task_config, tags + ['hpo'], container)
    # pprint.pprint(trial_task.to_dict(report=True))
    configurator.observe([trial_task.to_dict(report=True)])

    # TODO: We should remove this, there is normally only one create_trial task at a time for a
    #       given set of tags.
    clean_duplicates(mahler_client, configurator, trial_task, tags)

    mahler_client.close()
    raise SignalInterruptTask('HPO not completed')

# Sample trials for n workers
# if not trials submitted interrupt
