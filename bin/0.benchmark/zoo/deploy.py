import asyncio
import argparse
import os

from orion.core.io.experiment_builder import ExperimentBuilder

from sgdad.utils.commandline import execute


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT = "benchmark"

options = "array=1-10;mem=30000M;time=2:59:00"

flow_template = "flow-submit {container} --config {file_path} --options '{options}'{optionals}"

orion_template = "orion hunt -n '{experiment};{dataset};{model};{version}' --config /repos/sgd-space/configs/0.benchmark/orion_config.yaml"

commandline_template = "{flow} launch {orion}"

# orion save -n $experiment.$dataset.$model \
#     orion run --tags $experiment;$dataset;$model \
#         python src/sgdad/train.py --config=${file_path} --model-seed 1 --sampler-seed 1 --epochs 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--container', required=True, help='Container to use for the execution')
    parser.add_argument('--orion-config', type=argparse.FileType('r'),
                        help="custom configuration for orion")
    parser.add_argument('--configs', default='configs', help='Root folder for configs')
    parser.add_argument('--version', required=True, help='Version of the execution')
    parser.add_argument('--datasets', nargs="*", help='Datasets to save executions for')
    parser.add_argument('--models', nargs="*", help='Models to save executions for')
    parser.add_argument('--data-wrapper-levels', nargs="*", help='Noise levels to try')
    # TODO remove print_only, and turn it into a test for orion, if not using
    # orion to register this execution, then print-only
    parser.add_argument('--print-only', action='store_true',
                        help='Print commands but do not execute.')
    parser.add_argument('--generate-only', action='store_true',
                        help='Generate sbatch scripts but do not submit.')

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

            basedir = os.path.join(ROOT_DIR, 'submit', dataset)

            if not os.path.isdir(basedir):
                os.makedirs(basedir)

            file_path =  os.path.join(basedir, model + ".sh")

            yield dataset, model, file_path


def main(argv=None):

    args = parse_args(argv)

    iterator = get_instances(args.configs, args.datasets, args.models, "0.benchmark")
    futures = []
    for dataset, model, file_path in iterator:
        name = '{experiment};{dataset};{model};{version}'.format(
            experiment=EXPERIMENT, dataset=dataset, model=model, version=args.version)

        experiment = ExperimentBuilder().build_view({'name': name})

        if experiment.is_done:
            continue

        flow = flow_template.format(
            file_path=file_path, container=args.container, options=options,
            optionals=" --generate-only" if args.generate_only else "")
        orion = orion_template.format(
            experiment=EXPERIMENT, dataset=dataset, model=model,
            version=args.version)
        commandline = commandline_template.format(flow=flow, orion=orion)
        futures.append(execute(commandline, print_only=args.print_only))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*futures))
    loop.close()


if __name__ == "__main__":
    main()
