import random

import json

from mahler.client import Client


mahler_client = Client()      
db_client = mahler_client.registrar._db._db

VERSION_TAGS = ['v4.2.3', 'v4.2.4', 'v4.2.5', 'v4.2.6',
                'v5.0.0']

N_POINTS = 10

colors = """\
#8dd3c7
#d3d394
#bebada
#fb8072
#80b1d3
#fdb462
#b3de69
#fccde5
#d9d9d9
#bc80bd
#ccebc5
#ffed6f""".split("\n")

MODELS = ['lenet', 'mobilenetv2', 'vgg11', 'vgg19', 'densenet121', 'densenet201', 'resnet18', 'resnet101']
DATASETS = ['mnist', 'cifar10', 'cifar100']

# points: 10
# models
# datasets
# colors (for models)
# data:
#   mnist:
#     lenet: [0.1, 0.3, ...]
#     mobilenetv2: [0.1, 0.3, ...]
#


base = dict(
    models=MODELS,
    datasets=DATASETS,
    colors=dict(zip(MODELS, colors)),
    data=dict())


def dump(fetch_fct):
    data = dict()
    for dataset_name in DATASETS:
        data[dataset_name] = dict()
        for model_name in MODELS:
            data[dataset_name][model_name] = fetch_fct(dataset_name, model_name)

    return data


def dump_seed():
    base['data'] = dump(fetch_seed)
    with open('seed.json', 'w') as f:
        f.write(json.dumps(base))


def dump_hpo():
    base['data'] = dump(fetch_hpo)
    with open('hpo.json', 'w') as f:
        f.write(json.dumps(base))


def fetch_seed(dataset_name, model_name):

    data = []

    for version in VERSION_TAGS:
        query = {}
        query['registry.tags'] = {'$all': [version, model_name, dataset_name, 'seed']}
        query['registry.status'] = 'Completed'
        projection = {'output.last.test.error_rate': 1}
        trials = db_client.tasks.report.find(query, projection=projection)
        for trial in trials:
            try:
                data.append(trial['output']['last']['test']['error_rate'])
                print('Adding true {:>20} {:>20}'.format(dataset_name, model_name))
            except KeyError:
                pass

            if len(data) >= N_POINTS:
                break

    while len(data) < N_POINTS:
        print('Adding fake {:>20} {:>20}'.format(dataset_name, model_name))
        data.append(random.random())

    return data


def fetch_hpo(dataset_name, model_name):
    data = []

    for version in VERSION_TAGS:
        query = {}
        query['registry.tags'] = {'$all': [version, model_name, dataset_name, 'distrib', 'min']}
        query['registry.status'] = 'Completed'
        projection = {'output.last.test.error_rate': 1}
        trials = db_client.tasks.report.find(query, projection=projection)
        for trial in trials:
            try:
                data.append(trial['output']['last']['test']['error_rate'])
                print('Adding true {:>20} {:>20}'.format(dataset_name, model_name))
            except KeyError:
                pass

            if len(data) >= N_POINTS:
                break

    while len(data) < N_POINTS:
        print('Adding fake {:>20} {:>20}'.format(dataset_name, model_name))
        data.append(random.random())

    return data


if __name__ == "__main__":
    print("    ---")
    print("    SEEDS")
    print("    ---")
    dump_seed()
    print("\n\n")
    print("    ---")
    print("    HPO")
    print("    ---")
    dump_hpo()
