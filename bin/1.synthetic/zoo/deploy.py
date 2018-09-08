import asyncio
import argparse
import os

from kleio.core.io.trial_builder import TrialBuilder
from kleio.core.trial import status

from sgdad.utils.commandline import execute


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT = "synthetic"

flow_template = "flow-submit {file_path} {container}"

kleio_template = "kleio run --config /config/kleio.core/kleio_config.yaml --tags '{experiment};{dataset};{model};{version}'"

commandline_template = "{flow} {kleio}"

# orion save -n $experiment.$dataset.$model \
#     kleio run --tags $experiment;$dataset;$model \
#         python src/sgdad/train.py --config=${file_path} --model-seed 1 --sampler-seed 1 --epochs 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--container', required=True, help='Container to use for the execution')
    parser.add_argument('--kleio-config', type=argparse.FileType('r'),
                        help="custom configuration for kleio")
    parser.add_argument('--configs', default='configs', help='Root folder for configs')
    parser.add_argument('--version', required=True, help='Version of the execution')
    parser.add_argument('--datasets', nargs="*", help='Datasets to save executions for')
    parser.add_argument('--models', nargs="*", help='Models to save executions for')
    parser.add_argument('--data-wrapper-levels', nargs="*", help='Noise levels to try')
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
                           in os.listdir(os.path.join(configs_root, experiment, dataset))
                           if model.split(".")[-1] == "yaml"]

        for model in possible_models:

            if models and model not in models:
                continue

            file_path = os.path.join(ROOT_DIR, 'submit', dataset, model + ".sh")

            yield dataset, model, file_path


def main(argv=None):

    args = parse_args(argv)

    iterator = get_instances(args.configs, args.datasets, args.models, "1.synthetic")
    futures = []
    for dataset, model, file_path in iterator:
        tags = [EXPERIMENT, dataset, model, args.version]
        database = TrialBuilder().build_database({'config': args.kleio_config})
        query = {
            'tags': {'$all': tags},
            'registry.status': {'$in': status.RESERVABLE}
            }

        total = database.count('trials.reports', {'tags': {'$all': tags}})
        runnable = database.count('trials.reports', query)

        print()
        print("{:>40} Total:    {:>5}".format(";".join(tags[1:]), total))
        print("{:>40} Runnable: {:>5}".format("", runnable))

        if not runnable:
            continue

        flow = flow_template.format(file_path=file_path, container=args.container)
        kleio = kleio_template.format(
            experiment=EXPERIMENT, dataset=dataset, model=model,
            version=args.version)
        commandline = commandline_template.format(flow=flow, kleio=kleio)
        # flow-submit file_path kleio run --tags {experiment};{dataset};{model};{version}
        futures.append(execute(commandline, print_only=args.print_only))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*futures))
    loop.close()


if __name__ == "__main__":
    main()
