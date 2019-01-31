from datetime import datetime
import argparse
import pprint

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.timing import Timer
from ignite.metrics import Accuracy, Loss

import torch
import torch.nn.functional as F
import torch.optim

import yaml

from repro.dataset.base import build_dataset
from repro.model.base import (
    build_model, get_checkpoint_file_path, load_checkpoint, save_checkpoint, clear_checkpoint)
from repro.optimizer.base import build_optimizer
from repro.utils.flatten import unflatten, merge_configs


TIME_BUFFER = 60 * 5  # 5 minute


def update(config, arguments):
    pairs = [argument.split("=") for argument in arguments]
    kwargs = unflatten(dict((pair[0], eval(pair[1])) for pair in pairs))
    return merge_configs(config, kwargs)


def build_config(**kwargs):
    with open(kwargs.pop('config'), 'r') as f:
        config = yaml.load(f)

    if 'updates' in kwargs:
        if isinstance(kwargs['updates'], str):
            kwargs['updates'] = [kwargs['updates']]

        update(config, kwargs.pop('updates'))

    config.update(kwargs)

    return config


def build_experiment(**config):

    seeds = {'model': config.get('model_seed'),
             'sampler': config.get('sampler_seed')}

    if seeds['model'] is None:
        raise ValueError("model_seed must be defined")

    if seeds['sampler'] is None:
        raise ValueError("sampler_seed must be defined")

    device = torch.device('cpu')
    if torch.cuda.is_available():
        print("\n\nUsing GPU\n\n")
        device = torch.device('cuda')
    else:
        raise RuntimeError("GPU not available")

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

    lr_scheduler_config = config['optimizer'].pop("lr_scheduler", None)

    optimizer = build_optimizer(model=model, **config['optimizer'])

    if lr_scheduler_config:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=lr_scheduler_config['patience'])
    else:
        lr_scheduler = None

    print("\n\nOptimizer\n")
    pprint.pprint(optimizer)
    print("\n\n")

    return dataset, model, optimizer, lr_scheduler, device, seeds


def build_evaluators(trainer, model, device, patience, compute_test_error_rates):
    def build_evaluator():
        return create_supervised_evaluator(
            model, metrics={'accuracy': Accuracy(),
                            'nll': Loss(F.cross_entropy)},
            device=device)

    evaluators = dict(train=build_evaluator(), valid=build_evaluator())
    if compute_test_error_rates:
        evaluators['test'] = build_evaluator()

    def score_function(engine):
        return engine.state.metrics['accuracy']

    early_stopping_handler = EarlyStopping(patience=patience, score_function=score_function,
                                           trainer=trainer)
    evaluators['valid'].add_event_handler(Events.COMPLETED, early_stopping_handler)

    return evaluators


def main(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--config', help='Path to yaml configuration file for the trial')
    parser.add_argument('--model-seed', type=int, help='Seed for model\'s initialization')
    parser.add_argument('--sampler-seed', type=int, help='Seed for data sampling order')
    # parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train.')

    parser.add_argument('--updates', nargs='+', default=[], metavar='updates',
                        help='Values to update in the configuration file')

    args = parser.parse_args(argv)

    config = build_config(**vars(args))

    train(**config)


def train(data, model, optimizer, model_seed=1, sampler_seed=1, max_epochs=200,
          patience=None, compute_test_error_rates=False, loading_file_path=None):

    # Checkpointing file path is named based on Mahler task ID
    checkpointing_file_path = get_checkpoint_file_path()

    if loading_file_path is None:
        loading_file_path = checkpointing_file_path
    # Else, we are branching from another configuration.

    print("\n\nLoading file path:")
    print(loading_file_path)

    print("\n\nCheckpointing file path:")
    print(checkpointing_file_path)
    print("\n\n")

    dataset, model, optimizer, lr_scheduler, device, seeds = build_experiment(
        data=data, model=model, optimizer=optimizer,
        model_seed=model_seed, sampler_seed=sampler_seed)

    if lr_scheduler is None and patience is None:
        patience = 10
    elif patience is None:
        patience = lr_scheduler.patience * 2

    print("\n\nEarly stopping with patience: {}\n\n".format(patience))

    timer = Timer(average=True)

    trainer = create_supervised_trainer(
        model, optimizer, torch.nn.functional.cross_entropy, device=device)

    evaluators = build_evaluators(trainer, model, device, patience, compute_test_error_rates)

    timer.attach(trainer, start=Events.STARTED, step=Events.EPOCH_COMPLETED)

    all_stats = []
    best_stats = {}

    @trainer.on(Events.STARTED)
    def trainer_load_checkpoint(engine):
        engine.state.last_checkpoint = datetime.utcnow()
        metadata = load_checkpoint(loading_file_path, model, optimizer, lr_scheduler)
        if metadata:
            print('Resuming from epoch {}'.format(metadata['epoch']))
            engine.state.epoch = metadata['epoch']
            engine.state.iteration = metadata['iteration']
            for epoch_stats in metadata['all_stats']:
                all_stats.append(epoch_stats)
                if (not best_stats or
                        epoch_stats['valid']['error_rate'] < best_stats['valid']['error_rate']):
                    best_stats.update(epoch_stats)
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

        stats = dict(epoch=engine.state.epoch)

        for name, evaluator in evaluators.items():
            loader = dataset[name]
            metrics = evaluator.run(loader).metrics
            stats[name] = dict(
                loss=metrics['nll'],
                error_rate=1. - metrics['accuracy'])

        if lr_scheduler:
            lr_scheduler.step(stats['valid']['error_rate'])

        if not best_stats or stats['valid']['error_rate'] < best_stats['valid']['error_rate']:
            best_stats.update(stats)

        all_stats.append(stats)

        print(("Epoch {:>4} Iteration {:>12} Loss {:>8.3f} "
               "Best-Valid-ER {:>8.4f} Time {:>8.3f}").format(
            engine.state.epoch, engine.state.iteration, engine.state.output,
            best_stats['valid']['error_rate'], timer.value()))

        # TODO: Checkpoint lr_scheduler as well
        if (datetime.utcnow() - engine.state.last_checkpoint).total_seconds() > TIME_BUFFER:
            print('Checkpointing epoch {}'.format(engine.state.epoch))
            save_checkpoint(checkpointing_file_path,
                            model, optimizer, lr_scheduler,
                            epoch=engine.state.epoch,
                            iteration=engine.state.iteration,
                            all_stats=all_stats)
            engine.state.last_checkpoint = datetime.utcnow()

    print("Training")
    trainer.run(dataset['train'], max_epochs=max_epochs)

    # Remove checkpoint to avoid cluttering the FS.
    clear_checkpoint(checkpointing_file_path)

    return {'best': best_stats, 'all': tuple(all_stats)}


def seed(seed):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # random.seed(seed)
    # numpy.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
