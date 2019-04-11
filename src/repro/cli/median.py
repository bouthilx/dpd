import argparse
import itertools
import sys

import mahler.client as mahler

from repro.median import run, create_trial


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
        '--container', help='Container to execute HPO')
    parser.add_argument(
        '--datasets', default=tuple(), choices=DATASET_NAMES, type=str, nargs='*',
        help='Dataset to run')
    parser.add_argument(
        '--models', default=tuple(), choices=MODEL_NAMES, type=str, nargs='*',
        help='Models to run')
    parser.add_argument(
        '--max-epochs', default=120, type=int,
        help='Maximum number of epochs to train. May be stopped before by median or early '
             'stopping rules.')
    parser.add_argument(
        '--n-trials', default=2000, type=int,
        help='Number of trials for the global random search with median stopping rule.')
    parser.add_argument(
        '--n-bootstraps', default=10, type=int, help='Number of sub random search.')
    parser.add_argument(
        '--n-data-sampling', default=10, type=int,
        help='Number of data sampling per random search.')
    parser.add_argument(
        '--n-var-sampling', default=10, type=int,
        help='Number of init+data-order sampling per data sampling.')
    parser.add_argument(
        '--seed', default=1, type=int,
        help='Base seed for the entire process.')
    parser.add_argument(
        '--force', action='store_true', default=False,
        help='Register even if another similar task already exists')
    parser.add_argument(
        '--stopping-grace', default=5, type=int,
        help='Min epochs before starting median stopping')
    parser.add_argument(
        '--final-population', default=3, type=int,
        help='Min trials at epoch t to consider median stopping')
    parser.add_argument(
        '--stopping-n-steps', default=30, type=int,
        help='Number of steps where trials are stopped')
    parser.add_argument(
        '--stopping-window-size', default=11, type=int,
        help='Window width to smooth curves.')
    parser.add_argument(
        '--config-dir-path',
        default=DEFAULT_CONFIG_DIR_PATH,
        help=('Path of directory containing the configurations of the datasets and models. '
              'Default: {}').format(DEFAULT_CONFIG_DIR_PATH))

    parser.add_argument(
        '--debug', action='store_true', default=False, help='Deploy small setup for debugging.')

    options = parser.parse_args(argv)

    # TODO: Verify that n_boostrap is a divider of max_trials
    if options.n_trials % options.n_bootstraps != 0:
        print("Error: n-bootstrap ({}) is not a divider of n-trials ({})".format(
            options.n_trials, options.n_bootstraps))
        sys.exit(2)

    mahler_client = mahler.Client()

    # for i in range(options.num_workers):
    setup_combinations = itertools.product(
        options.datasets, options.models)
    for dataset_name, model_name in setup_combinations:
        tags = options.tags + [dataset_name, model_name, 'repro', 'median']

        if any(True for _ in mahler_client.find(tags=tags + [run.name])) and not options.force:
            print('already registered for tags: {}'.format(", ".join(tags)))
            continue

        print("Registering {} with tags: {}".format(create_trial.name, ", ".join(tags)))

        mahler_client.register(
            create_trial.delay(
                config_dir_path=options.config_dir_path,
                dataset_name=dataset_name,
                model_name=model_name,
                max_epochs=options.max_epochs,
                n_trials=options.n_trials,
                stopping_rule=dict(
                    initial_population=options.n_trials,
                    final_population=options.final_population,
                    n_steps=options.stopping_n_steps,
                    window_size=options.stopping_window_size,
                    n_points=options.max_epochs),
                bootstrap=dict(
                    n_bootstraps=options.n_bootstraps,
                    n_data_sampling=options.n_data_sampling,
                    n_var_sampling=options.n_var_sampling),
                seed=options.seed),
            container=options.container, tags=tags)


if __name__ == "__main__":
    main()
