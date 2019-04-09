import copy
import logging
import os
import random
import time
import sys

import numpy

import mahler.client as mahler
from mahler.core.utils.flatten import flatten, unflatten
from mahler.core.utils.errors import SignalSuspend

from orion.core.io.space_builder import Space, DimensionBuilder

import yaml

from repro.train import train


logger = logging.getLogger(__name__)


USAGE = {
    'lenet': {
        'gpu': {
            'memory': 2**30,  # 1 GB
            'util': 10}},
    'mobilenetv2': {
        'gpu': {
            'memory': 3 * 2**30,  # 1 GB
            'util': 40}},
    'vgg11': {
        'gpu': {
            'memory': 3 * 2**30,  # 1 GB
            'util': 40}},
    'vgg19': {
        'gpu': {
            'memory': 4 * 2**30,  # 1 GB
            'util': 60}},
    'resnet18': {
        'gpu': {
            'memory': 3 * 2**30,  # 1 GB
            'util': 60}},
    'resnet101': {
        'gpu': {
            'memory': 4 * 2**30,  # 1 GB
            'util': 60}},
    'else': {
        'gpu': {
            'memory': 4 * 2 ** 30,  # 1 GB
            'util': 50}}}


run = mahler.operator(resources={'cpu': 4, 'gpu': 1, 'mem': '20GB'}, resumable=True)(train)

# run needs
# data, model, optimizer, model_seed, sampler_seed, max_epochs

# This is build from config_file + args of create_trila


def load_config(config_dir_path, dataset_name, model_name):

    config_path = os.path.join(config_dir_path, "{dataset}/{model}.yaml").format(
        dataset=dataset_name, model=model_name)

    with open(config_path, 'r') as f:
        config = yaml.load(f)

    return config


def merge(config, subconfig):
    flattened_config = copy.deepcopy(flatten(config))
    flattened_config.update(flatten(subconfig))
    return unflatten(flattened_config)


def register_new_trial(mahler_client, trial_config, tags, container):
    trial_config = copy.deepcopy(trial_config)
    trial_config['model_seed'] = random.uniform(1, 10000)
    trial_config['sampler_seed'] = random.uniform(1, 10000)
    model_usage = USAGE[trial_config['model']['name']]
    new_task = mahler_client.register(
        run.delay(**trial_config),
        resources={'usage': model_usage}, container=container, tags=tags)
    print(new_task.id, sorted(new_task.tags))
    return new_task


def sample_new_config(space, config, hpo_seed, global_seed):
    params = unflatten(dict(zip(space.keys(), space.sample(seed=hpo_seed)[0])))
    params['optimizer']['lr_scheduler'] = dict(patience=10)
    params['model_seed'] = global_seed
    params['sampler_seed'] = global_seed
    # Params can be all
    new_config = merge(config, params)
    new_config['data']['seed'] = global_seed
    return new_config


# TODO: Support resources properly
#       (should submit based on largest request and enable running small tasks in large resource
#        workers)
@mahler.operator(resources={'cpu': 1, 'mem': '1GB', 'usage': {'gpu': {'memory': 0, 'util': 0}}})
def create_trial(config_dir_path, dataset_name, model_name,
                 grace, min_population,
                 max_epochs, n_trials, bootstrap, seed):

    usage = USAGE[model_name]

    # Create client inside function otherwise MongoDB does not play nicely with multiprocessing
    mahler_client = mahler.Client()

    task = mahler_client.get_current_task()
    tags = [tag for tag in task.tags if tag != task.name]
    container = task.container

    projection = {'output': 1, 'arguments': 1, 'registry.status': 1}  # , 'bounds.priority': 1}
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
    config['grace'] = grace
    config['min_population'] = min_population

    n_broken = 0
    prev_trials = list(mahler_client.find(tags=tags + [run.name, 'hpo'], _return_doc=True,
                                          _projection=projection))

    trials = []

    n_uncompleted = 0
    n_completed = 0

    for trial in prev_trials:
        if trial['registry']['status'] == 'Cancelled':
            continue

        n_broken += int(trial['registry']['status'] == 'Completed' and not trial['output'])
        n_broken += int(trial['registry']['status'] == 'Broken')
        # Not Completed, or Completed but broken with empty output
        if trial['registry']['status'] != 'Completed' or not trial['output']:
            print(trial['id'], trial['registry']['status'])
            n_uncompleted += 1
        else:
            n_completed += 1

        task = mahler_client._create_shallow_task(trial['id'])
        task._arguments = trial['arguments']
        trials.append(task)

        # TODO: Turn trials doc into task objects.

    if n_broken > 20:
        message = (
            '{} trials are broken. Suspending creation of trials until investigation '
            'is done.').format(n_broken)

        mahler_client.close()
        raise SignalSuspend(message)

    seeds = numpy.random.RandomState(seed).randint(1, 1000000, size=n_trials)

    print("Generating new configurations")
    for i in range(max(n_trials - len(trials), 0)):
        config['max_epochs'] = max_epochs
        new_task_config = sample_new_config(
            space, config, hpo_seed=int(seeds[len(trials) - 1]), global_seed=seed)
        new_task = run.delay(**new_task_config)
        mahler_client.register(new_task, container=container, tags=tags + ['hpo'],
                               resources={'usage': usage})
        trials.append(new_task)

    # draw bootstrap samples, if any is all completed, register seeds
    assert len(trials) == n_trials

    while not draw_bootstrap_samples(mahler_client, tags, container, trials, seed, **bootstrap):
        print('Waiting 5 mins for trials to complete')
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(60 * 5)

    mahler_client.close()


# TODO: Pass arguments here, adapt the max boostraps and other max for cli/hpo
#       x Add medianstopping criterion after 20 runs, otherwise to much resource wasted
#       x Add metrics
#       x Test metrics
#       x Implement base multiprocess for tests
#       x Test base multiprocessing
#       x Add multiprocess handling in workers for better resource usage.
#       x Test multiprocessing
#       Test data shuffling
#       Test reproducibility
#       Test priority
#       See if we can wrap loader with transforms
#       Add command to rsync data locally beforce executing workers
#       Make maintainer only fetch recently updated tasks, avoid fetching all old ones

#       For later:
#       Add option to use f16 precision
#           - Verify that it is reproducible
#           - Compare with f32 results

def draw_bootstrap_samples(mahler_client, tags, container, trials, seed, n_bootstraps,
                           n_data_sampling, n_var_sampling):
    # From trials, sample batches
    # For each batch, test if all trials completed, if not, skip
    # if batch seeding < max_seed: sample
    # if all batches seeding >= max_seed: return True

    samples = []
    n_trials = len(trials)
    bootstrap_size = int(n_trials / n_bootstraps)

    indices = list(range(n_trials))
    rng = numpy.random.RandomState(seed)
    rng.shuffle(indices)
    for i in range(0, n_trials, bootstrap_size):
        sample = dict(
            indexes=indices[i:i + bootstrap_size],
            seed=rng.randint(1, 1000000))
        assert len(sample['indexes']) == bootstrap_size
        samples.append(sample)

    completed = True
    for batch_index, bootstrap_sample in enumerate(samples):
        skip = False
        # batch_priority = -(n_bootstraps - batch_index)
        bootstrap_trials = [trials[index] for index in bootstrap_sample['indexes']]
        for trial in bootstrap_trials:
            if trial.status.name not in ['Completed', 'Suspended']:
                skip = True
                # Increase priority (the smaller, the highest priority, like `nice`)
                # if trial.priority > batch_priority:
                #     mahler_client.update_priority(trial, batch_priority)

        if skip:
            logger.debug('Bootstrap sample {} have non completed trials'.format(batch_index))
            completed = False
            continue

        params = get_best_params(
            mahler_client, batch_index, bootstrap_trials)

        params['compute_test_error_rates'] = True

        memory = 0
        total_mem_used = 0
        util = 0
        for usage in bootstrap_trials[0].metrics['usage']:
            if usage['gpu']['memory']['process']['max'] > memory:
                memory = usage['gpu']['memory']['process']['max']
                total_mem_used = usage['gpu']['memory']['used']['max']
                util = usage['gpu']['util']['mean']

        if total_mem_used == 0.:
            util = 0
        else:
            util = int(memory / total_mem_used * util + 0.5)

        usage = {'gpu': {'memory': memory, 'util': util}}

        print('Generating trials for bootstrap {}'.format(batch_index))
        sample_distrib(
            mahler_client, tags, container, batch_index, seed, params, usage,
            n_data_sampling, n_var_sampling)

    return completed


# Config
# config_dir_path:
# dataset_name:
# model_name:
# max_epochs:
# n_trials: 500
# bootstrap:
#     samples: 10
#     seed: 1
#     distrib:
#         data_sampling: 10
#         var_sampling: 10


def get_best_params(mahler_client, batch_index, trials):

    best_trial = None
    best_objective = float('inf')

    for trial in trials:
        tag = 'b-{}'.format(batch_index)
        # Check beforehand to avoid fetching the tags for testing
        if tag not in trial.tags:
            mahler_client.add_tags(trial, [tag])

        if trial.status.name == 'Suspended':
            continue

        objective = trial.output['best']['valid']['error_rate']
        if objective < best_objective:
            best_objective = objective
            best_trial = trial

    if best_trial:
        return best_trial.arguments

    return None


def sample_distrib(mahler_client, tags, container, batch_index, seed, params, usage,
                   n_data_sampling, n_var_sampling):
    # if batch seeding < max_seed: sample
    # if all batches seeding >= max_seed: return True

    batch_tags = ['distrib', 'b-{}'.format(batch_index)]

    projection = {'registry.tags': 1, 'arguments.model_seed': 1, 'arguments.data.seed': 1}

    # All distribs for this batch
    trials = mahler_client.find(tags=tags + batch_tags + [run.name],
                                _return_doc=True, _projection=projection)

    # Build a dict of all trials already registered.
    distrib = dict()
    for trial in trials:
        # TODO: Sort by data_sampling, init_sampling
        trial['registry']['tags']
        trial['arguments']['model_seed']
        trial['arguments']['data']['seed']

        # for each data sampling
        # x times model init and data order seeded
        # Use the same so we can compare a model on different shuffling with same init and data
        # order.
        data_sampling_seed = trial['arguments']['data']['seed']
        if data_sampling_seed not in distrib:
            distrib[data_sampling_seed] = {}

        distrib[data_sampling_seed][trial['arguments']['model_seed']] = trial

    # We want the same seeds across all HPO batches, so that they can be compared point to point
    # across HPOs
    rng = numpy.random.RandomState(seed)
    data_seeds = [seed] + list(rng.randint(1, 1000000, size=n_data_sampling - 1))
    var_seeds = rng.randint(1, 1000000, size=n_var_sampling)
    for data_seed in data_seeds:
        data_seed = int(data_seed)
        if data_seed not in distrib:
            distrib[data_seed] = {}
        for var_seed in var_seeds:
            var_seed = int(var_seed)
            if var_seed in distrib[data_seed]:
                continue

            params['sampler_seed'] = var_seed
            params['model_seed'] = var_seed
            params['data']['seed'] = data_seed

            # priority = -(max_var_sampling - j)
            distrib[data_seed][var_seed] = mahler_client.register(
                run.delay(**params), container=container, tags=tags + batch_tags,
                resources={'usage': usage})
            #     priority=priority)

    mahler_client.close()
