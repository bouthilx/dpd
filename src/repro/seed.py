import copy
import random
import os

import mahler.client as mahler

import yaml

from repro.train import train


run = mahler.operator(resources={'cpu': 4, 'gpu': 1, 'mem': '20BG'}, resumable=True)(train)


def create_trial(mahler_client, config_dir_path, dataset_name, model_name, tags, container=None):
    config = load_config(config_dir_path, dataset_name, model_name)

    new_trial_tasks = []
    for i in range(10):
        trial_task = register_new_trial(mahler_client, config, tags + ['seed'], container)
        new_trial_tasks.append(trial_task)
        print('Registered task {}'.format(trial_task.id))


def register_new_trial(mahler_client, config, tags, container):
    config = copy.deepcopy(config)
    config['model_seed'] = random.uniform(1, 10000)
    config['sampler_seed'] = random.uniform(1, 10000)
    # Milestones from from https://arxiv.org/pdf/1603.05027.pdf
    print(config['optimizer'])
    config['max_epochs'] = 120
    config['optimizer']['lr_scheduler'] = dict(milestones=[1, 30, 60, 120])
    return mahler_client.register(run.delay(**config), container=container, tags=tags)


def load_config(config_dir_path, dataset_name, model_name):

    config_path = os.path.join(config_dir_path, "{dataset}/{model}.yaml").format(
        dataset=dataset_name, model=model_name)

    with open(config_path, 'r') as f:
        config = yaml.load(f)

    return config



