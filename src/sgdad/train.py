import argparse
import pprint

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy

from orion.client import report_results

import torch

import yaml

from sgdad.dataset.base import build_dataset
from sgdad.model.base import build_model, load_model, save_model
from sgdad.optimizer.base import build_optimizer


def main(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--config', help='Path to yaml configuration file for the trial')
    parser.add_argument('--model-seed', type=int, required=True, help='Seed for model\'s initialization')
    parser.add_argument('--sampler-seed', type=int, required=True, help='Seed for data sampling order')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train.')

    args = parser.parse_args(argv)

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        print("\n\nUsing GPU\n\n")
        device = torch.device('cuda')
    else:
        print("\n\nUsing CPU\n\n")

    print("\n\nConfiguration\n")
    pprint.pprint(config)
    print("\n\n")

    seed(args.sampler_seed)

    dataset = build_dataset(**config['data'])
    train_loader = dataset['train']
    valid_loader = dataset['valid']
    input_size = dataset['input_size']
    num_classes = dataset['num_classes']

    print("\n\nDatasets\n")
    pprint.pprint(dataset)
    print("\n\n")

    # Note: model is not loaded here for resumed trials
    seed(args.model_seed)
    model = build_model(input_size=input_size, num_classes=num_classes, **config['model'])

    print("\n\nModel\n")
    pprint.pprint(model)
    print("\n\n")

    optimizer = build_optimizer(model=model, **config['optimizer'])

    print("\n\nOptimizer\n")
    pprint.pprint(optimizer)
    print("\n\n")

    trainer = create_supervised_trainer(
        model, optimizer, torch.nn.functional.nll_loss, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={'accuracy': CategoricalAccuracy()}, device=device)

    @trainer.on(Events.STARTED)
    def trainer_load_model(engine):
        metadata = load_model(model, 'model')
        if metadata:
            engine.state.epoch = metadata['epoch']
        else:
            save_model(model, 'model', epoch=0)

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_save_model(engine):
        print("Epoch {:>4} Iteration {:>8}".format(engine.state.epoch, engine.state.iteration))
        save_model(model, 'model', epoch=engine.state.epoch)

    trainer.run(train_loader, max_epochs=args.epochs)

    evaluator.run(valid_loader)
    accuracy = evaluator.state.metrics['accuracy']
    report_results([dict(
        name="valid_error_rate",
        type="objective",
        value= 1.0 - accuracy)])


def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
