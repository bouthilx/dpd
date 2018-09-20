import argparse
import pprint
import random

import numpy

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy

from kleio.core.io.resolve_config import merge_configs
from kleio.core.utils import unflatten

from kleio.client.logger import kleio_logger

from orion.client import report_results

import torch

import yaml
from sgdad.dataset.base import build_dataset
from sgdad.model.base import build_model, load_model, save_model
from sgdad.optimizer.base import build_optimizer


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

    if isinstance(kwargs['updates'] , str):
        kwargs['updates'] = [kwargs['updates']]

    update(config, kwargs['updates'])

    device = torch.device('cpu')
    if torch.cuda.is_available():
        print("\n\nUsing GPU\n\n")
        device = torch.device('cuda')
    else:
        print("\n\nUsing CPU\n\n")

    print("\n\nConfiguration\n")
    pprint.pprint(config)
    print("\n\n")

    seed(int(kwargs['sampler_seed']))

    dataset = build_dataset(**config['data'])
    input_size = dataset['input_size']
    num_classes = dataset['num_classes']

    print("\n\nDatasets\n")
    pprint.pprint(dataset)
    print("\n\n")

    # Note: model is not loaded here for resumed trials
    seed(int(kwargs['model_seed']))
    model = build_model(input_size=input_size, num_classes=num_classes, **config['model'])

    print("\n\nModel\n")
    pprint.pprint(model)
    print("\n\n")

    optimizer = build_optimizer(model=model, **config['optimizer'])

    print("\n\nOptimizer\n")
    pprint.pprint(optimizer)
    print("\n\n")

    return dataset, model, optimizer, device


def main(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--config', help='Path to yaml configuration file for the trial')
    parser.add_argument('--model-seed', type=int, required=True, help='Seed for model\'s initialization')
    parser.add_argument('--sampler-seed', type=int, required=True, help='Seed for data sampling order')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train.')

    parser.add_argument('--updates', nargs='+', default=[], metavar='updates',
                        help='Values to update in the configuration file')

    args = parser.parse_args(argv)

    dataset, model, optimizer, device = build_experiment(**vars(args))

    train_loader = dataset['train']
    valid_loader = dataset['valid']
    test_loader = dataset['test']

    trainer = create_supervised_trainer(
        model, optimizer, torch.nn.functional.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={'accuracy': CategoricalAccuracy()}, device=device)

    @trainer.on(Events.STARTED)
    def trainer_load_model(engine):
        metadata = load_model(model, 'model')
        if metadata:
            engine.state.epoch = metadata['epoch']
        else:
            save_model(model, 'model', epoch=0)

    @trainer.on(Events.EPOCH_STARTED)
    def trainer_seeding(engine):
        seed(args.sampler_seed + engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_save_model(engine):
        train_acc = evaluator.run(train_loader).metrics['accuracy']
        valid_acc = evaluator.run(valid_loader).metrics['accuracy']
        test_acc = evaluator.run(test_loader).metrics['accuracy']

        kleio_logger.log_statistic(**{
            'epoch': engine.state.epoch,
            'train': dict(
                loss=engine.state.output,
                error_rate=1. - train_acc
            ),
            'valid': dict(
                error_rate=1. - valid_acc
            ),
            'test': dict(
                error_rate=1. - test_acc
            ),
        })

        print("Epoch {:>4} Iteration {:>8} Loss {:>12}".format(engine.state.epoch, engine.state.iteration, engine.state.output))
        save_model(model, 'model', epoch=engine.state.epoch)

    print("Training")
    trainer.run(train_loader, max_epochs=args.epochs)

    evaluator.run(valid_loader)
    accuracy = evaluator.state.metrics['accuracy']
    report_results([dict(
        name="valid_error_rate",
        type="objective",
        value= 1.0 - accuracy)])


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
