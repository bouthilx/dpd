from collections import OrderedDict
import argparse
import pprint
import sys

from kleio.client.logger import kleio_logger
from kleio.client.logger import AnalyzeLogger
from kleio.core.utils import flatten, unflatten

import torch

from sgdad.train import build_experiment, update, seed
from sgdad.dataset.base import build_dataset, build_wrapper
from sgdad.dataset.wrapper import infinite
from sgdad.model.base import build_model, load_model, save_model
from sgdad.optimizer.base import build_optimizer
from sgdad.analysis.base import build_analysis
from sgdad.utils.yaml import ordered_load


##
## Multiple datasets? Like original + spherical cow
## Multiple analysis?
##


def analysis_already_exists(trial, query):
    if kleio_logger.trial is None:
        return False

    tags = ";".join(kleio_logger.trial.tags)

    tags_statistics = trial.statistics.tags

    if tags not in tags_statistics.keys():
        return False

    return query['epoch'] in tags_statistics[tags].epoch.keys()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--config', help='Path to yaml configuration file for the trial')
    parser.add_argument('--trial-id', help='ID of the trial to analyze')
    parser.add_argument('--updates', nargs='+', default=[], metavar='updates',
                        help='Values to update in the configuration file')

    args = parser.parse_args(argv)

    with open(args.config, 'r') as f:
        config = ordered_load(f)

    update(config, args.updates)

    device = torch.device("cuda")

    print("\n\nRunning analysis on trial {}\n".format(args.trial_id))

    if kleio_logger.trial is None:
        print("Analysis executed without Kleio, results will be printed but not logged")
    else:
        print("Analysis id is {}\n".format(kleio_logger.trial.id))

    trial_logger = AnalyzeLogger(args.trial_id)
    trial_args = trial_logger.load_config()

    print("\n\nTrial logged configuration:\n")

    pprint.pprint(trial_args)

    # Force num_workers to 0
    if 'num_workers' in trial_args['config']['content']['data']:
        print("Forced num_workers to 0 for reproducibility")
        trial_args['config']['content']['data']['num_workers'] = 0

    # Check if statistic already logged with current tag, if yes, quit.
    # For that, look for query[epoch] and ";".join(kleio_logger.trial.tags)
    if analysis_already_exists(trial_logger.trial, config['query']):
        print("Analyze tagged '{}' for the following query already exists.\n{}".format(
            ";".join(kleio_logger.trial.tags), pprint.pformat(config['query'])))
        print("\nLeaving without recomputing it...")
        sys.exit(0)

    dataset, model, optimizer, device = build_experiment(**trial_args)

    # Fetch after build_experiment, because it does update content based on trial_args[updates]
    trial_config = trial_args['config']['content']

    print("\n\nQuerying from\n")
    pprint.pprint(config['query'])
    print("\n\n")

    model.to(device)

    metadata = load_model(model, 'model', config['query'], trial_logger)
    assert metadata is not None
    runtime_timestamp = metadata['runtime_timestamp']

    print("\n\nCorresponding snapshot's metadata\n")
    pprint.pprint(metadata)
    print("\n\n")

    print("\n\nData used for the analyses\n")
    pprint.pprint(config['data'])
    print("\n\n")

    seed(int(trial_args['sampler_seed']) + config['query']['epoch'])

    loaders = OrderedDict()
    pump_out_n_batches = config['data'].get('pump_out', None)
    for name, data_config in config['data'].items():
        print("Preparing {}".format(name))
        if name == "model":
            model_data = OrderedDict()
            for set_name in data_config['select']:
                model_data[set_name] = pump_out(dataset[set_name], pump_out_n_batches)
            loaders[trial_config['data']['name']] = model_data
            continue

        select = data_config.pop('select')
        if 'name' not in data_config:
            data_config['name'] = name
        analyze_dataset = build_wrapper(dataset, **data_config)
        analyze_data = OrderedDict()
        for set_name in select:
            # Iterate now to keep same order throughout analyses
            analyze_data[set_name] = pump_out(analyze_dataset[set_name], pump_out_n_batches)
        loaders[name] = analyze_data

    # print("Limiting all datasets to the size of the smallest one, "
    #       "giving {} mini-batches".format(number_of_batches))

    # for name, dataset in loaders.items():
    #     for set_name, loader in dataset.items():
    #         dataset[set_name] = loader[:number_of_batches]

    print("\n\nAnalyses\n")
    pprint.pprint(config['analyses'])
    print("\n\n")

    analyses = OrderedDict()
    for analysis_config in config['analyses']:
        if isinstance(analysis_config, str):
            name = analysis_config
            analysis_config = OrderedDict()
        else:
            name, analysis_config = next(iter(analysis_config.items()))
        analysis_config['name'] = name
        analyses[name] = build_analysis(**analysis_config)

    statistics = OrderedDict()
    for name, dataset in loaders.items():
        results = OrderedDict()
        for set_name, loader in dataset.items():
            results[set_name] = OrderedDict()
            for analysis_name, analysis in analyses.items():
                print("Computing {} on {}.{}".format(analysis_name, name, set_name))
                results[set_name].update(
                    analysis(results[set_name], name, set_name, loader, model, optimizer, device))

        statistics[name] = results

    statistics = curate(statistics)
    trial_logger.insert_statistic(
        timestamp=runtime_timestamp, epoch=config['query']['epoch'], **statistics)


def curate(d):
    curated = {}
    for name, dataset in d.items():
        for set_name, loader in dataset.items():
            for key, value in flatten(d[name][set_name], returncopy=False).items():
                if is_hidden(key):
                    continue

                curated["{}.{}.{}".format(name, set_name, curate_key(key))] = value

    return unflatten(curated)


def is_hidden(key):
    return any(subkey.startswith("_") for subkey in key.split("."))


def curate_key(key):
    return ".".join([name for name in key.split(".") if not is_hidden(name)])


def pump_out(loader, number_of_batches):
    if number_of_batches is None:
        # TODO: Make this generic to adapt to any datasets
        return [batch for batch in loader][:78]  # Smaller sets in MNIST and CIFAR10

    batches = []
    for i, batch in enumerate(infinite.extend(loader)):
        if i >= number_of_batches:
            break
        batches.append(batch)

    return batches


if __name__ == "__main__":
    main()
