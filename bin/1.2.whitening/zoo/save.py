import asyncio
import argparse
import os

from sgdad.utils.commandline import execute


EXPERIMENT = "whitening"

kleio_template = 'kleio save --branch-original --config /config/kleio.core/kleio_config.yaml --tags {experiment};{dataset};{model};{version}'

script_template = (
    "python3.6 /repos/sgd-space/src/sgdad/train.py --config={file_path} "
    "--model-seed 1 --sampler-seed 1 --epochs 300 "
    "--update data.whitening_level={data_whitening_level}")

commandline_template = "{kleio} {script}"

# orion save -n $experiment.$dataset.$model \
#     kleio run --tags $experiment;$dataset;$model \
#         python src/sgdad/train.py --config=${file_path} --model-seed 1 --sampler-seed 1 --epochs 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--configs', default='configs', help='Root folder for configs')
    parser.add_argument('--version', required=True, help='Version of the execution')
    parser.add_argument('--datasets', nargs="*", help='Datasets to save executions for')
    parser.add_argument('--models', nargs="*", help='Models to save executions for')
    parser.add_argument('--data-whitening-levels', nargs="*", help='Noise levels to try')
    # TODO remove print_only, and turn it into a test for kleio, if not using
    # kleio to register this execution, then print-only
    parser.add_argument('--print-only', action='store_true',
                        help='Print executions but do not execute.')

    return parser.parse_args(argv)


def get_instances(configs_root, datasets, models, experiment):
    possible_datasets = [dataset for dataset
                         in os.listdir(os.path.join(configs_root, experiment))
                         if os.path.isdir(os.path.join(configs_root, experiment, dataset))]

    for dataset in possible_datasets:

        if datasets and dataset not in datasets:
            continue

        possible_models = [model[:-5] for model 
                           in os.listdir(os.path.join(configs_root, experiment, 'zoo', dataset))
                           if model.split(".")[-1] == "yaml"]

        for model in possible_models:

            if models and model not in models:
                continue

            file_path = os.path.join(configs_root, experiment, dataset, model + ".yaml")

            yield dataset, model, file_path


def main(argv=None):

    args = parse_args(argv)

    data_whitening_levels = args.data_whitening_levels
    if not data_whitening_levels:
        data_whitening_levels = [i / 10. for i in range(0, 11)]

    iterator = get_instances(args.configs, args.datasets, args.models, "1.2.whitening")
    futures = []
    for dataset, model, file_path in iterator:
        for data_whitening_level in data_whitening_levels:
            kleio = kleio_template.format(
                experiment=EXPERIMENT, dataset=dataset, model=model, version=args.version)
            script = script_template.format(
                file_path=file_path, data_whitening_level=data_whitening_level)
            commandline = commandline_template.format(kleio=kleio, script=script)
            futures.append(execute(commandline, print_only=args.print_only))
            if len(futures) > 10:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(asyncio.gather(*futures))
                futures = []

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*futures))
    loop.close()


if __name__ == "__main__":
    main()
