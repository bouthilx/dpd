from datetime import datetime
import argparse
import pprint
import random
import numpy

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.timing import Timer
from ignite.metrics import Metric, Loss

import torch
import torch.nn.functional as F
import torch.optim

import yaml

from repro.dataset.base import build_dataset
from repro.log import Logger
from repro.model.base import (
    build_model, get_checkpoint_file_path, load_checkpoint, save_checkpoint, clear_checkpoint)
from repro.optimizer.base import build_optimizer
from repro.utils.flatten import unflatten, merge_configs


TIME_BUFFER = 30


class ErrorRate(Metric):
    def reset(self):
        self._n = 0
        self._m = 0
        self._mu = 0

    def update(self, output):
        y_pred, y = output

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred "
                             "must have shape of (batch_size, num_classes, ...) or "
                             "(batch_size, ...).")

        if y.ndimension() > 1 and y.shape[1] == 1:
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=1)

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0], ) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if y_pred.ndimension() == y.ndimension():
            # Maps Binary Case to Categorical Case with 2 classes
            y_pred = y_pred.unsqueeze(dim=1)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

        indices = torch.max(y_pred, dim=1)[1]
        error = 1 - torch.eq(indices, y).view(-1)

        n = self._n
        m = self._m
        mean = self._mu

        for x in error:
            x = float(x)
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            m += delta * delta2

            # mu_nt = mu_n + (x - mu_n) / n
            # m = m + (x - mu_n) * (x - mu_nt)
            # mu_n = mu_nt

        self._m = m
        self._mu = mean
        self._n = n

    def compute(self):
        if self._n == 2:
            raise RuntimeError('ErrorRate must have at least two example before it can be computed')

        return {'mean': self._mu, 'var': self._m / (self._n - 1)}


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
            optimizer, mode='min', patience=lr_scheduler_config['patience'],
            factor=lr_scheduler_config['factor'], verbose=True)
    else:
        lr_scheduler = None

    print("\n\nOptimizer\n")
    pprint.pprint(optimizer)
    print("\n\n")

    return dataset, model, optimizer, lr_scheduler, device, seeds


def build_evaluators(trainer, model, device, patience, compute_test_error_rates):
    def build_evaluator():
        return create_supervised_evaluator(
            model, metrics={'error_rate': ErrorRate(),
                            'nll': Loss(F.cross_entropy)},
            device=device)

    evaluators = dict(train=build_evaluator(), valid=build_evaluator())
    if compute_test_error_rates:
        evaluators['test'] = build_evaluator()

    def score_function(engine):
        return 1 - engine.state.metrics['error_rate']['mean']

    early_stopping_handler = EarlyStopping(patience=patience, score_function=score_function,
                                           trainer=trainer)
    evaluators['valid'].add_event_handler(Events.COMPLETED, early_stopping_handler)

    return evaluators, early_stopping_handler


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


def train(data, model, optimizer, model_seed=1, sampler_seed=1, max_epochs=120,
          patience=None, stopping_rule=None,
          compute_test_error_rates=False, loading_file_path=None, callback=None):

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
        patience = 20
    elif patience is None:
        patience = lr_scheduler.patience * 2

    print("\n\nMax epochs: {}\n\n".format(max_epochs))

    print("\n\nEarly stopping with patience: {}\n\n".format(patience))

    print('Building timers, training and evaluation loops...')
    timer = Timer(average=True)

    print('    Stopping timer')
    stopping_timer = Timer(average=True)

    print('    Training loop')
    trainer = create_supervised_trainer(
        model, optimizer, torch.nn.functional.cross_entropy, device=device)

    print('    Evaluator loop')
    evaluators, early_stopping = build_evaluators(trainer, model, device, patience, compute_test_error_rates)

    print('    Set timer events')
    timer.attach(trainer, start=Events.STARTED, step=Events.EPOCH_COMPLETED)

    print('    Metric logger')
    metric_logger = Logger()
    print('Done')

    all_stats = []
    best_stats = {}

    @trainer.on(Events.STARTED)
    def trainer_load_checkpoint(engine):
        engine.state.last_checkpoint = datetime.utcnow()
        metadata = load_checkpoint(loading_file_path, model, optimizer, lr_scheduler)
        if metadata:
            print('Resuming from epoch {}'.format(metadata['epoch']))
            print('Optimizer:')
            print('    lr:', optimizer.param_groups[0]['lr'])
            print('    momentum:', optimizer.param_groups[0]['momentum'])
            print('    weight decay:', optimizer.param_groups[0]['weight_decay'])

            print('LR schedule:')
            print('    best:', lr_scheduler.best)
            print('    num_bad_epochs:', lr_scheduler.num_bad_epochs)
            print('    cooldown:', lr_scheduler.cooldown)

            engine.state.epoch = metadata['epoch']
            engine.state.iteration = metadata['iteration']
            for epoch_stats in metadata['all_stats']:
                tmp = engine.state.metrics
                engine.state.metrics = epoch_stats['valid']
                early_stopping(engine)
                engine.state.metrics = tmp

                all_stats.append(epoch_stats)
                if (not best_stats or
                        (epoch_stats['valid']['error_rate']['mean'] <
                         best_stats['valid']['error_rate']['mean'])):
                    best_stats.update(epoch_stats)

            print('Early stopping:')
            print('    best_score:', early_stopping.best_score)
            print('    counter:', early_stopping.counter)
        else:
            engine.state.epoch = 0
            engine.state.iteration = 0
            engine.state.output = 0.0
            # trainer_save_checkpoint(engine)

    @trainer.on(Events.EPOCH_STARTED)
    def trainer_seeding(engine):
        print(seeds['sampler'] + engine.state.epoch)
        seed(int(seeds['sampler'] + engine.state.epoch))
        model.train()

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_save_checkpoint(engine):
        model.eval()

        stats = dict(epoch=engine.state.epoch)

        for name in ['valid', 'train', 'test']:
            evaluator = evaluators.get(name, None)
            if evaluator is None:
                continue

            loader = dataset[name]
            metrics = evaluator.run(loader).metrics
            stats[name] = dict(
                loss=metrics['nll'],
                error_rate= metrics['error_rate'])

        print('Early stopping')
        print('{}   {} < {}'.format(early_stopping.best_score, early_stopping.counter,
                                    early_stopping.patience))

        current_v_error_rate = stats['valid']['error_rate']['mean']
        best_v_error_rate = best_stats.get('valid', {}).get('error_rate', {}).get('mean', 100)

        if lr_scheduler:
            lr_scheduler.step(current_v_error_rate)
            print('Lr schedule')
            print('{}   last_epoch: {} bads: {} cooldown: {}'.format(
                lr_scheduler.best, lr_scheduler.last_epoch, lr_scheduler.num_bad_epochs, lr_scheduler.cooldown_counter))

        if not best_stats or current_v_error_rate < best_v_error_rate:
            best_stats.update(stats)

        # TODO: load all tasks with the same tags in mahler, compute the error_rate at that point
        #       (compare median of best error_rates up to that point vs this best_stats
        #       if below median, suspend
        #       maybe, interrupt and increase priority, or not... Because we would need to wait for
        #       it to completed anyway
        #       Grace period? Like 60 epochs? :/
        #       Or reduce quantile as time grows (stop worst 95th quantile at 10 epochs, 50th at
        #       100, 75th at 150 and so on...) Meh to much novelty.
        #       min trials at that point?
        #       or interrupt after each 10/20 epochs, so that number of trials is quickly high
        #       but that means we need a way to log results during execution, not just output.

        print(("Epoch {:>4} Iteration {:>12} Loss {:>8.3f} "
               "Best-Valid-ER {:>8.4f} Time {:>8.3f}").format(
            engine.state.epoch, engine.state.iteration, engine.state.output, best_v_error_rate,
            timer.value()))

        metric_logger.add_metric(stats)

        all_stats.append(stats)

        # TODO: Checkpoint lr_scheduler as well
        if (datetime.utcnow() - engine.state.last_checkpoint).total_seconds() > TIME_BUFFER:
            print('Checkpointing epoch {}'.format(engine.state.epoch))
            save_checkpoint(checkpointing_file_path,
                            model, optimizer, lr_scheduler,
                            epoch=engine.state.epoch,
                            iteration=engine.state.iteration,
                            all_stats=all_stats)
            engine.state.last_checkpoint = datetime.utcnow()

        if callback:
            callback(step=engine.state.epoch, objective=stats['valid']['error_rate']['mean'],
                     finished=False)

    print("Training")
    trainer.run(dataset['train'], max_epochs=max_epochs)

    metric_logger.close()

    # Remove checkpoint to avoid cluttering the FS.
    clear_checkpoint(checkpointing_file_path)

    if callback:
        callback(step=max_epochs, objective=all_stats[-1]['valid']['error_rate']['mean'],
                 finished=True)

    return {'best': best_stats, 'all': tuple(all_stats)}


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
