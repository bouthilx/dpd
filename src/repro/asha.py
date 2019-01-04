import os
import random

from mahler.client import Client
from mahler.core.utils.flatten import flatten, unflatten

from orion.core.io.space_builder import Space, DimensionBuilder

import yaml

from repro.train import train
from repro.hpo.asha import ASHA


space = Space()
space['optimizer.lr'] = (
    DimensionBuilder().build('optimizer.lr', 'loguniform(1e-5, 0.5)'))
space['optimizer.lr_scheduler.milestones'] = (
    DimensionBuilder().build('optimizer.lr_scheduler.milestones', 'uniform(0, 1, shape=(4, ))'))
space['optimizer.momentum'] = (
    DimensionBuilder().build('optimizer.momentum', 'uniform(0., 0.9)'))
space['optimizer.weight_decay'] = (
    DimensionBuilder().build('optimizer.weight_decay', 'loguniform(1e-8, 1e-3)'))


def convert_params(params):
    # Not samples from space but arguments of registered task, no need to convert
    if "data" not in params:
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

mahler = Client()

run = mahler.operator(resources={'cpu': 4, 'gpu': 1, 'mem': '20BG'}, resumable=True)(train)


# run needs
# data, model, optimizer, model_seed, sampler_seed, max_epochs

# This is build from config_file + args of create_trila


def load_config(config_dir_path, dataset_name, model_name):

    print(config_dir_path)
    print(dataset_name, model_name)
    print(os.path.join(config_dir_path, "{dataset}/{model}.yaml"))
    config_path = os.path.join(config_dir_path, "{dataset}/{model}.yaml").format(
        dataset=dataset_name, model=model_name)

    with open(config_path, 'r') as f:
        config = yaml.load(f)

    return config


# TODO: Support resources properly
#       (should submit based on largest request and enable running small tasks in large resource
#        workers)
#@mahler.operator(resources={'cpu':1, 'mem':'1GB'})
@mahler.operator(resources={'cpu':4, 'gpu': 1, 'mem':'20GB'})
def create_trial(config_dir_path, dataset_name, model_name, asha_config):

    config = load_config(config_dir_path, dataset_name, model_name)
    config['model_seed'] = random.uniform(1, 10000)
    config['sampler_seed'] = random.uniform(1, 10000)

    task = mahler.get_task()
    tags = [tag for tag in task.tags if tag != task.name]
    container = task.container

    trials = mahler.find(tags=tags + [run.name])

    asha = ASHA(space, dict(max_epochs=FIDELITY_LEVELS), **asha_config)
    asha.observe(trials)

    if asha.final_rung_is_filled():
        return
    
    params = asha.get_params()
    params = convert_params(params)
    # Params can be all 
    flattened_config = flatten(config)
    flattened_config.update(flatten(params))
    config = unflatten(flattened_config)

    trial_task = mahler.register(run.delay(**config), container=container, tags=tags)
    create_task = mahler.register(
        create_trial.delay(config_dir_path=config_dir_path, dataset_name=dataset_name,
                           model_name=model_name, asha_config=asha_config),
        container=container, tags=tags)

    return dict(trial_task_id=str(trial_task.id),
                create_trial_task_id=str(create_task.id))
