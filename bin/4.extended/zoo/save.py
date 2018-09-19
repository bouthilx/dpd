import argparse
import asyncio
import os

from sgdad.utils.commandline import execute


EXPERIMENT = "restricted"

script_template = ("python src/sgdad/train.py --config={file_path} "
                   "--model-seed {model_seed} --sampler-seed {sampler_seed} --epochs 100")

def create_save_cmd(dataset, model, version, file_path, model_seed, sampler_seed):
    if model_seed == 1 and sampler_seed == 1:
        return create_insert_trial_cmd(dataset, model, version, file_path,
                                       model_seed, sampler_seed)
    else:
        return create_save_trial_cmd(dataset, model, version, file_path,
                                     model_seed, sampler_seed)


def create_insert_trial_cmd(dataset, model, version, file_path, model_seed, sampler_seed, print_only=False):
    orion_template = "orion insert -n extended.{dataset}.{model}.{version}"
    kleio_template = "kleio run --tags extended;restricted;{dataset};{model};{version}"
    commandline_template = "{orion} {kleio} {script}"

    orion = orion_template.format(dataset=dataset, model=model, version=version)
    kleio = kleio_template.format(
        experiment=EXPERIMENT, dataset=dataset, model=model, version=version)
    script = script_template.format(
        file_path=file_path, model_seed=model_seed, sampler_seed=sampler_seed)

    return commandline_template.format(orion=orion, kleio=kleio, script=script)


def create_save_trial_cmd(dataset, model, version, file_path, model_seed, sampler_seed):
    kleio_template = "kleio save --tags restricted;{dataset};{model};{version}"
    commandline_template = "{kleio} {script}"

    kleio = kleio_template.format(
        experiment=EXPERIMENT, dataset=dataset, model=model, version=version)
    script = script_template.format(
        file_path=file_path, model_seed=model_seed, sampler_seed=sampler_seed)

    return commandline_template.format(kleio=kleio, script=script)


# orion save -n $experiment.$dataset.$model \
#      kleio run --tags $experiment;$dataset;$model \
#          python src/sgdad/train.py --config=${file_path} --model-seed 1 --sampler-seed 1 --epochs 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--configs', default='configs', help='Root folder for configs')
    parser.add_argument('--version', required=True, help='Version of the execution')
    parser.add_argument('--datasets', nargs="*", help='Datasets to save executions for')
    parser.add_argument('--models', nargs="*", help='Models to save executions for')
    parser.add_argument('--model-seeds', nargs="*", type=int, help='Seeds for model initialization')
    parser.add_argument('--sampler-seeds', nargs="*", type=int, help='Seeds for data sampler')
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


def get_seeds(model_seeds, sampler_seeds):
    if not model_seeds:
        model_seeds = list(range(1, 21))

    if not sampler_seeds:
        sampler_seeds = list(range(1, 21))

    seeds = set()
    for model_seed in model_seeds:
        point = (model_seed, 1)
        if point not in seeds:
            seeds.add(point)
            yield point

    for sampler_seed in sampler_seeds:
        point = (1, sampler_seed)
        if point not in seeds:
            seeds.add(point)
            yield point

    
def main(argv=None):

    args = parse_args(argv)

    iterator = get_instances(args.configs, args.datasets, args.models, "2.restricted")
    futures = []
    for dataset, model, file_path in iterator:
        for model_seed, sampler_seed in get_seeds(args.model_seeds, args.sampler_seeds):
            commandline = create_save_cmd(dataset, model, args.version, file_path,
                                          model_seed, sampler_seed)
            futures.append(execute(commandline, print_only=args.print_only))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*futures))
    loop.close()


if __name__ == "__main__":
    main()
