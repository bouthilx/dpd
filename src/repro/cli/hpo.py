import argparse
import itertools
import math

import mahler.client as mahler

from repro.asha import run, create_trial


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

# asha_config = dict(name='asha', reduction_factor=4, max_resource=MAX_RESOURCE,
#                    fidelity_space=dict(max_epochs=[15, 30, 60]))

asha_config = dict(name='asha', reduction_factor=1, max_resource=MAX_RESOURCE,
                   fidelity_space=dict(max_epochs=[1, 2, 4]))

random_search_config = dict(name='random_search',
                            max_trials=total_trials(MAX_EPOCHS, **asha_config))

configurator_configs = dict(asha=asha_config, random_search=random_search_config)



def main(argv=None):
    # NOTE: When implementing full pipeline, config will become dynamic and change based on which
    # (dataset, model) pair to run
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument(
        '--tags', nargs='*', type=str, required=True,
        help=('Tags for the tasks. '
              'Note: must have the format (tag1|tag2|...) to be compatible with singularity'))
    parser.add_argument(
        '--container', help='Container to execute HPO')
    parser.add_argument(
        '--datasets', default=tuple(), choices=DATASET_NAMES, type=str, nargs='*',
        help='Dataset to run')
    parser.add_argument(
        '--configurators', default=tuple(), choices=configurator_configs.keys(), type=str,
        nargs='*', help='Configurators to run')
    parser.add_argument(
        '--models', default=tuple(), choices=MODEL_NAMES, type=str, nargs='*',
        help='Models to run')
    parser.add_argument(
        '--force', action='store_true', default=False,
        help='Register even if another similar task already exists')
    parser.add_argument(
        '--config-dir-path',
        default=DEFAULT_CONFIG_DIR_PATH,
        help=('Path of directory containing the configurations of the datasets and models. '
              'Default: {}').format(DEFAULT_CONFIG_DIR_PATH))

    options = parser.parse_args(argv)

    mahler_client = mahler.Client()

    # for i in range(options.num_workers):
    setup_combinations = itertools.product(
        options.datasets, options.models, options.configurators)
    for dataset_name, model_name, configurator_name in setup_combinations:
        tags = options.tags + [dataset_name, model_name, 'repro', configurator_name]

        if any(True for _ in mahler_client.find(tags=tags + [run.name])) and not options.force:
            print('HPO `{}` already registered for tags: {}'.format(
                configurator_name, ", ".join(tags)))
            continue

        print("Registering {} with tags: {}".format(create_trial.name, ", ".join(tags)))

        configurator_config = configurator_configs[configurator_name]

        mahler_client.register(
            create_trial.delay(
                config_dir_path=options.config_dir_path,
                dataset_name=dataset_name,
                model_name=model_name,
                configurator_config=configurator_config,
                max_epochs=MAX_EPOCHS,
                max_workers=MAX_WORKERS,
                max_resource=MAX_RESOURCE,
                number_of_seeds=NUMBER_OF_SEEDS),
            container=options.container, tags=tags)


if __name__ == "__main__":
    main()
