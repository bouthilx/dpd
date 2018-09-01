import argparse
import asyncio
import os

from sgdad.utils.commandline import execute


EXPERIMENT = "restricted"

orion_template = "orion init_only -n extended.{dataset}.{model}.{version}"
kleio_template = "kleio run --tags extended;restricted;{dataset};{model};{version}"
script_template = ("python src/sgdad/train.py --config={file_path} "
                   "--model-seed 1 --sampler-seed 1")

commandline_template = "{orion} {kleio} {script}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--configs', default='configs', help='Root folder for configs')
    parser.add_argument('--version', required=True, help='Version of the execution')
    parser.add_argument('--datasets', nargs="*", help='Datasets to save executions for')
    parser.add_argument('--models', nargs="*", help='Models to save executions for')
    parser.add_argument('--print-only', action='store_true',
                        help='Print executions but do not execute.')

    return parser.parse_args(argv)


def get_instances(configs_root, datasets, models, experiment):
    possible_datasets = os.listdir(os.path.join(configs_root, experiment))

    for dataset in possible_datasets:

        if datasets and dataset not in datasets:
            continue

        possible_models = [model[:-5] for model 
                           in os.listdir(os.path.join(configs_root, experiment, dataset))
                           if model.split(".")[-1] == "yaml"]

        for model in possible_models:

            if models and model not in models:
                continue

            file_path = os.path.join(configs_root, experiment, dataset, model + ".yaml")

            yield dataset, model, file_path

    
def main(argv=None):

    args = parse_args(argv)

    iterator = get_instances(args.configs, args.datasets, args.models, "3.extended")
    futures = []
    for dataset, model, file_path in iterator:
        orion = orion_template.format(dataset=dataset, model=model, version=args.version)
        kleio = kleio_template.format(
            experiment=EXPERIMENT, dataset=dataset, model=model, version=args.version)
        script = script_template.format(file_path=file_path)

        commandline = commandline_template.format(orion=orion, kleio=kleio, script=script)

        print(commandline)
        futures.append(execute(commandline, print_only=args.print_only))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*futures))
    loop.close()


if __name__ == "__main__":
    main()
