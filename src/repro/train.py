import argparse
import pprint
import random

import numpy

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss

from kleio.core.io.resolve_config import merge_configs
from kleio.core.utils import unflatten

from kleio.client.logger import kleio_logger

from orion.client import report_results

import torch
import torch.nn.functional as F

import yaml
from repro.dataset.base import build_dataset
from repro.model.base import build_model, load_checkpoint, save_checkpoint
from repro.optimizer.base import build_optimizer


EPOCS_TO_SAVE = list(range(6)) + [10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300]


def update(config, arguments):
    pairs = [argument.split("=") for argument in arguments]
    kwargs = unflatten(dict((pair[0], eval(pair[1])) for pair in pairs))
    return merge_configs(config, kwargs)


def build_experiment(**kwargs):

    if isinstance(kwargs['config'], str):
        with open(kwargs['config'], 'r') as f:
            config = yaml.load(f)
    else:
        config = kwargs['config']['content']

    if 'update' in kwargs:
        kwargs['updates'] = kwargs['update']

    if 'updates' in kwargs:
        if isinstance(kwargs['updates'], str):
            kwargs['updates'] = [kwargs['updates']]

        update(config, kwargs['updates'])

    seeds = {'model': config.get('model_seed', kwargs.get('model_seed', None)),
             'sampler': config.get('sampler_seed', kwargs.get('sampler_seed', None))}

    if seeds['model'] is None:
        raise ValueError("model_seed must be defined")

    if seeds['sampler'] is None:
        raise ValueError("sampler_seed must be defined")

    device = torch.device('cpu')
    if torch.cuda.is_available():
        print("\n\nUsing GPU\n\n")
        device = torch.device('cuda')
    else:
        print("\n\nUsing CPU\n\n")

    print("\n\nConfiguration\n")
    pprint.pprint(config)
    print("\n\n")

    seed(int(seeds['sampler']))

    dataset = build_dataset(**config['data'])
    input_size = dataset['input_size']
    num_classes = dataset['num_classes']

    print("\n\nDatasets\n")
    pprint.pprint(dataset)
    print("\n\n")

    # Note: model is not loaded here for resumed trials
    seed(int(seeds['model']))
    model = build_model(input_size=input_size, num_classes=num_classes, **config['model'])

    print("\n\nModel\n")
    pprint.pprint(model)
    print("\n\n")

    optimizer = build_optimizer(model=model, **config['optimizer'])

    print("\n\nOptimizer\n")
    pprint.pprint(optimizer)
    print("\n\n")

    return dataset, model, optimizer, device, seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--config', help='Path to yaml configuration file for the trial')
    parser.add_argument('--model-seed', type=int, help='Seed for model\'s initialization')
    parser.add_argument('--sampler-seed', type=int, help='Seed for data sampling order')
    # parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train.')

    parser.add_argument('--updates', nargs='+', default=[], metavar='updates',
                        help='Values to update in the configuration file')

    args = parser.parse_args(argv)

    dataset, model, optimizer, device, seeds = build_experiment(**vars(args))

    train_loader = dataset['train']
    valid_loader = dataset['valid']
    test_loader = dataset['test']

    trainer = create_supervised_trainer(
        model, optimizer, torch.nn.functional.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={'accuracy': CategoricalAccuracy(),
                        'nll': Loss(F.cross_entropy)},
        device=device)

    @trainer.on(Events.STARTED)
    def trainer_load_checkpoint(engine):
        metadata = load_checkpoint(model, optimizer, 'checkpoint')
        if metadata:
            engine.state.epoch = metadata['epoch']
            engine.state.iteration = metadata['iteration']
        else:
            engine.state.epoch = 0
            engine.state.iteration = 0
            engine.state.output = 0.0
            trainer_save_checkpoint(engine)

    @trainer.on(Events.EPOCH_STARTED)
    def trainer_seeding(engine):
        seed(seeds['sampler'] + engine.state.epoch)
        model.train()

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_save_checkpoint(engine):
        model.eval()
        train_metrics = evaluator.run(train_loader).metrics
        valid_metrics = evaluator.run(valid_loader).metrics
        test_metrics = evaluator.run(test_loader).metrics

        kleio_logger.log_statistic(**{
            'epoch': engine.state.epoch,
            'train': dict(
                loss=train_metrics['nll'],
                error_rate=1. - train_metrics['accuracy']
            ),
            'valid': dict(
                loss=valid_metrics['nll'],
                error_rate=1. - valid_metrics['accuracy']
            ),
            'test': dict(
                loss=test_metrics['nll'],
                error_rate=1. - test_metrics['accuracy']
            ),
        })

        print("Epoch {:>4} Iteration {:>8} Loss {:>12}".format(
            engine.state.epoch, engine.state.iteration, engine.state.output))
        if engine.state.epoch in EPOCS_TO_SAVE:
            save_checkpoint(model, optimizer, 'checkpoint',
                            epoch=engine.state.epoch,
                            iteration=engine.state.iteration)

    print("Training")
    trainer.run(train_loader, max_epochs=300)

    evaluator.run(valid_loader)
    accuracy = evaluator.state.metrics['accuracy']
    report_results([dict(
        name="valid_error_rate",
        type="objective",
        value=1.0 - accuracy)])


def seed(seed):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()