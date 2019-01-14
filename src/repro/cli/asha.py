import argparse

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


def main(argv=None):
    # NOTE: When implementing full pipeline, config will become dynamic and change based on which
    # (dataset, model) pair to run
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument(
        '--tags', nargs='*', type=str, required=True,
        help=('Tags for the tasks. '
              'Note: must have the format (tag1|tag2|...) to be compatible with singularity'))
    parser.add_argument(
        '--num-workers', required=True, type=int, help='Number of workers for ASHA')
    parser.add_argument(
        '--reduction-factor', required=True, type=int, help='Reduction factor for ASHA')
    parser.add_argument(
        '--max-resource', required=True, type=int,
        help='Number of trials to execute with full budget')
    parser.add_argument(
        '--container', help='Container to execute HPO')
    parser.add_argument(
        '--datasets', default=tuple(), choices=DATASET_NAMES, type=str, nargs='*',
        help='Dataset to run')
    parser.add_argument(
        '--models', default=tuple(), choices=MODEL_NAMES, type=str, nargs='*',
        help='Models to run')
    parser.add_argument(
        '--config-dir-path',
        default=DEFAULT_CONFIG_DIR_PATH,
        help=('Path of directory containing the configurations of the datasets and models. '
              'Default: {}').format(DEFAULT_CONFIG_DIR_PATH))

    options = parser.parse_args(argv)

    mahler_client = mahler.Client()

    # for i in range(options.num_workers):
    for dataset_name in options.datasets:
        for model_name in options.models:
            tags = options.tags + [dataset_name, model_name, 'repro']

            if any(True for _ in mahler_client.find(tags=tags + [run.name])):
                print('HPO already registered for tags: {}'.format(", ".join(tags)))
                continue

            print("Registering {} with tags: {}".format(create_trial.name, ", ".join(tags)))

            mahler_client.register(
                create_trial.delay(
                    config_dir_path=options.config_dir_path,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    asha_config=dict(reduction_factor=options.reduction_factor,
                                     max_resource=options.max_resource)),
                container=options.container, tags=tags)


if __name__ == "__main__":
    main()
