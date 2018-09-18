import asyncio
import argparse
import os

from kleio.core.io.trial_builder import TrialBuilder
from kleio.core.trial import status

from sgdad.utils.commandline import execute


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT = "synthetic"

options_template = "{array}mem=30000M;time=2:59:00"

flow_template = "flow-submit {container} --config {file_path} --options {options}{optionals}"

kleio_template = """\
kleio run --allow-host-change \
--config /config/kleio.core/kleio_config.yaml \
--tags {experiment};{dataset};{model};{execution_version};{analysis_version}\
"""

commandline_template = "{flow} launch {kleio}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--container', required=True, help='Container to use for the execution')
    parser.add_argument('--kleio-config', type=argparse.FileType('r'),
                        help="custom configuration for kleio")
    parser.add_argument('--configs', default='configs', help='Root folder for configs')
    parser.add_argument('--execution-version', required=True, help='Version of the execution')
    parser.add_argument('--analysis-version', required=True, help='Version of the analysis')
    parser.add_argument('--datasets', nargs="*", help='Datasets to save executions for')
    parser.add_argument('--models', nargs="*", help='Models to save executions for')
    # TODO remove print_only, and turn it into a test for kleio, if not using
    # kleio to register this execution, then print-only
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

            file_path = os.path.join(basedir, model + ".sh")

            yield dataset, model, file_path


def main(argv=None):

    args = parse_args(argv)

    iterator = get_instances(args.configs, args.datasets, args.models, "1.synthetic")
    futures = []

    database = TrialBuilder().build_database({'config': args.kleio_config})
    for dataset, model, file_path in iterator:
        tags = [EXPERIMENT, dataset, model, args.execution_version, args.analysis_version]
        query = {
            'tags': {'$all': tags},
            'registry.status': {'$in': status.RESERVABLE}}

        total = database.count('trials.reports', {'tags': {'$all': tags}})
        runnable = database.count('trials.reports', query)

        print()
        print("{:>40} Total:    {:>5}".format(";".join(tags[1:]), total))
        print("{:>40} Runnable: {:>5}".format("", runnable))

        if not runnable:
            continue

        if runnable > 20:
            options = options_template.format(array="array=1-{};".format(min(runnable // 10, 10)))
        else:
            options = options_template.format(array="")

        flow = flow_template.format(
            file_path=file_path, container=args.container, options=options,
            optionals=" --generate-only" if args.generate_only else "")
        kleio = kleio_template.format(
            experiment=EXPERIMENT, dataset=dataset, model=model,
            execution_version=args.execution_version, analysis_version=args.analysis_version)
        commandline = commandline_template.format(flow=flow, kleio=kleio)
        # flow-submit file_path kleio run --tags {experiment};{dataset};{model};{version}
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
