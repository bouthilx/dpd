import asyncio
import argparse
import os

from kleio.core.io.trial_builder import TrialBuilder

from sgdad.utils.commandline import execute


EXPERIMENT = "synthetic"

kleio_template = """\
kleio save --branch-original --config /config/kleio.core/kleio_config.yaml \
--tags {experiment};{dataset};{model};{analysis_version}"""

script_template = (
    "python3.6 /repos/sgd-space/src/sgdad/analyze.py "
    "--config=/repos/sgd-space/{file_path} --trial-id {trial_id} "
    "--updates query.epoch={epoch}")

commandline_template = "{kleio} {script}"


def analysis_already_exists(tags, trial_id, query):
    tags = ";".join(tags)

    # TODO: Build statistics directly to avoid loading the entire trial.
    trial = None
    tags_statistics = trial.statistics.tags

    if tags not in tags_statistics.keys():
        return False

    return query['epoch'] in tags_statistics[tags].epoch.keys()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--config', required=True, help='Configuration file for the analyses')
    parser.add_argument('--configs', default='configs', help='Root folder for configs')
    parser.add_argument('--kleio-config', type=argparse.FileType('r'),
                        default='/config/kleio.core/kleio_config.yaml',
                        help="custom configuration for kleio")
    parser.add_argument('--execution-version', required=True, help='Version of the execution')
    parser.add_argument('--analysis-version', required=True, help='Version of the analysis')
    parser.add_argument('--datasets', nargs="*", help='Datasets to save executions for')
    parser.add_argument('--models', nargs="*", help='Models to save executions for')
    parser.add_argument(
        '--epochs', type=str, required=True,
        help=('Epochs to analyze. Can be one epoch, or many separated by comas, or intervals '
              'separated by dot-columns. Ex: --epochs 300 or --epochs 100,200,300 or '
              '--epochs \'0:300:50\''))
    # TODO remove print_only, and turn it into a test for kleio, if not using
    # kleio to register this execution, then print-only
    parser.add_argument('--print-only', action='store_true',
                        help='Print executions but do not execute.')

    return parser.parse_args(argv)


def get_instances(configs_root, datasets, models, experiment):
    possible_datasets = [dataset for dataset
                         in os.listdir(os.path.join(configs_root, experiment, 'zoo'))
                         if os.path.isdir(os.path.join(configs_root, experiment, 'zoo', dataset))]

    for dataset in possible_datasets:

        if datasets and dataset not in datasets:
            continue

        possible_models = [model[:-5] for model
                           in os.listdir(os.path.join(configs_root, experiment, 'zoo', dataset))
                           if model.split(".")[-1] == "yaml"]

        for model in possible_models:

            if models and model not in models:
                continue

            yield dataset, model


def parse_epochs(string):
    if "," in string:
        return [int(e) for e in string.split(",")]

    if ":" in string:
        return list(range(*[int(e) for e in string.split(":")]))

    return [int(string)]


def fetch_trial_ids(database, dataset, model, version):
    tags = [EXPERIMENT, dataset, model, version]
    query = {
        '$and': [
            {'tags': {'$all': tags}},
            {'tags': {'$size': len(tags)}}
        ],
        'registry.status': {'$eq': 'completed'}}

    for trial_doc in database.read('trials.reports', query):
        yield trial_doc['_id']


def main(argv=None):

    args = parse_args(argv)

    database = TrialBuilder().build_database({'config': args.kleio_config})

    epochs = parse_epochs(args.epochs)
    # if not epochs:
    #     epochs = [i / 10. for i in range(0, 11)]

    iterator = get_instances(args.configs, args.datasets, args.models, "1.synthetic")
    futures = []
    for dataset, model in iterator:
        for trial_id in fetch_trial_ids(database, dataset, model, args.execution_version):
            for epoch in epochs:
                kleio = kleio_template.format(
                    experiment=EXPERIMENT, dataset=dataset, model=model,
                    analysis_version=args.analysis_version)
                script = script_template.format(
                    file_path=args.config, epoch=epoch, trial_id=trial_id)
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
