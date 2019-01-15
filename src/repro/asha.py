from collections import defaultdict
import copy
import logging
import os
import pprint
import random

import mahler.client as mahler
from mahler.core.utils.flatten import flatten, unflatten

from orion.core.io.space_builder import Space, DimensionBuilder

import yaml

from repro.train import train
from repro.hpo.asha import ASHA


logger = logging.getLogger(__name__)



def convert_params(params):
    # Not samples from space but arguments of registered task, no need to convert
    if isinstance(params['optimizer']['lr_scheduler']['milestones'], list):
        return params

    lr_schedule = params['optimizer']['lr_scheduler']['milestones']
    first_step = int(lr_schedule[0] >= 0.5)
    next_steps = ((lr_schedule[1:] / lr_schedule[1:].sum()).cumsum() * (200 - first_step) +
                  first_step)
    milestones = [int(step) for step in ([first_step] + list(next_steps))]
    params['optimizer']['lr_scheduler']['milestones'] = milestones

    return params


FIDELITY_LEVELS = [
    10,
    20,
    50,
    100,
    200]


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


def clean_duplicates(mahler_client, asha, new_tasks, tags):
    number_of_duplicates = 0
    number_of_duplicates = defaultdict(int)
    originals = dict()
    # TODO: Fetch documents rather than tasks, and use projection to just get the arguments
    #       This is all we need here.
    for task_document in mahler_client.find(tags=tags + [run.name], _return_doc=True,
                                            _projection={'arguments': 1, 'registry.status': 1}):
        task_arguments = task_document['arguments']
        task_id = task_document['id']

        for new_task in new_tasks:
            new_arguments = new_task.arguments
            new_id = new_task.id

            if task_arguments[asha.fidelity_dim] != new_arguments[asha.fidelity_dim]:
                continue

            if asha._fetch_trial_params(task_arguments) == asha._fetch_trial_params(new_arguments):
                number_of_duplicates[new_id] += 1

                if (number_of_duplicates[new_id] > 1 and
                        task_document['registry']['status'] != 'Cancelled'):
                    try:
                        message = 'Duplicate of task {}'.format(originals[new_id]['id'])
                        mahler_client.cancel(task_id, message)
                    except Exception as e:
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
    for trial in trials:
        objective = trial['output']['last']['valid']['error_rate']
        if min_objective > objective:
            min_objective = objective
            min_args = trial['arguments']

        if max_objective < objective:
            max_objective = objective
            max_args = trial['arguments']

        flattened_arguments = flatten(trial['arguments'])
        for key in space.keys():
            item = flattened_arguments[key]
            if key.endswith('milestones'):
                mean_args[key] = [item[i] + mean_args[key][i]
                                  if key in mean_args
                                  else item[i]
                                  for i in range(len(item))]
            else:
                mean_args[key] = item + mean_args.get(key, 0)

    for key, item in mean_args.items():
        if key.endswith('milestones'):
            mean_args[key] = [int(item_i / len(trials)) for item_i in mean_args[key]]
        else:
            mean_args[key] = item / len(trials)

    return min_args, max_args, mean_args


def register_best_trials(mahler_client, asha, tags, container):

    min_args, max_args, mean_args = compute_args(asha.rungs[len(FIDELITY_LEVELS) - 1], asha.space)

    # Use first trial in last rung, we don't care since they all have the same arguments
    # except for the optimizer HPs.
    config = asha.rungs[len(FIDELITY_LEVELS) - 1][0]['arguments']

    min_config = merge(config, min_args)
    max_config = merge(config, max_args)
    mean_config = merge(config, mean_args)

    new_trial_ids = defaultdict(list)
    for i in range(20):
        new_trial_ids['min'].append(
            register_new_trial(mahler_client, min_config, tags + ['distrib', 'min'], container).id)
        new_trial_ids['max'].append(
            register_new_trial(mahler_client, max_config, tags + ['distrib', 'max'], container).id)
        new_trial_ids['mean'].append(
            register_new_trial(mahler_client, mean_config, tags + ['distrib', 'mean'], container).id)

    return new_trial_ids


def merge(config, subconfig):
    flattened_config = copy.deepcopy(flatten(config))
    flattened_config.update(flatten(subconfig))
    return unflatten(flattened_config)
    

def register_new_trial(mahler_client, config, tags, container):
    config = copy.deepcopy(config)
    config['model_seed'] = random.uniform(1, 10000)
    config['sampler_seed'] = random.uniform(1, 10000)
    return mahler_client.register(run.delay(**config), container=container, tags=tags)


def sample_new_config(asha, config):
    params = asha.get_params()
    params = convert_params(params)
    # Params can be all 
    return merge(config, params)


# TODO: Support resources properly
#       (should submit based on largest request and enable running small tasks in large resource
#        workers)
#@mahler.operator(resources={'cpu':1, 'mem':'1GB'})
@mahler.operator(resources={'cpu':4, 'gpu': 1, 'mem':'20GB'})
def create_trial(config_dir_path, dataset_name, model_name, asha_config):

    # Create client inside function otherwise MongoDB does not play nicely with multiprocessing
    mahler_client = mahler.Client()

    # TODO: Will this work in multiprocessing? Maybe the Dispatcher will be a different object 
    #       because it is a different process.
    # NOTE: It seems to...
    task = mahler_client.get_current_task()
    tags = [tag for tag in task.tags if tag != task.name]
    container = task.container

    projection = {'output': 1, 'arguments': 1, 'registry.status': 1}
    trials = mahler_client.find(tags=tags + ['asha', run.name], _return_doc=True, _projection=projection)

    # *IMPORTANT* NOTE: 
    #     Space must be instantiated in the function otherwise its internal RNG state gets copied
    #     every time a task is forked in a process and each of them will start with the exact
    #     same state, leading to identical sampling. What a naughty side effect!
    space = Space()
    space['optimizer.lr'] = (
        DimensionBuilder().build('optimizer.lr', 'loguniform(1e-5, 0.5)'))
    space['optimizer.lr_scheduler.milestones'] = (
        DimensionBuilder().build('optimizer.lr_scheduler.milestones', 'uniform(0, 1, shape=(4, ))'))
    space['optimizer.momentum'] = (
        DimensionBuilder().build('optimizer.momentum', 'uniform(0., 0.9)'))
    space['optimizer.weight_decay'] = (
        DimensionBuilder().build('optimizer.weight_decay', 'loguniform(1e-8, 1e-3)'))

    config = load_config(config_dir_path, dataset_name, model_name)

    asha = ASHA(space, dict(max_epochs=FIDELITY_LEVELS), **asha_config)
    for trial in trials:
        if trial['registry']['status'] != 'Cancelled':
            asha.observe([trial])

    if asha.final_rung_is_filled():
        return register_best_trials(mahler_client, asha, tags, container)

    new_trial_tasks = []
    for i in range(2):  # 20):
        new_task_config = sample_new_config(asha, config)
        trial_task = register_new_trial(mahler_client, new_task_config, tags + ['asha'], container)
        # pprint.pprint(trial_task.to_dict(report=True))
        asha.observe([trial_task.to_dict(report=True)])
        new_trial_tasks.append(trial_task)

    clean_duplicates(mahler_client, asha, new_trial_tasks, tags)

    create_task = mahler_client.register(
        create_trial.delay(config_dir_path=config_dir_path, dataset_name=dataset_name,
                           model_name=model_name, asha_config=asha_config),
        container=container, tags=tags)

    return dict(trial_task_ids=[str(new_trial_task.id) for new_trial_task in new_trial_tasks],
                create_trial_task_id=str(create_task.id))
