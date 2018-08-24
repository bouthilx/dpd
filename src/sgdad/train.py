import argparse

import yaml

from sgdad.dataset.base import build_dataset
from sgdad.model.base import build_model, load_model, save_model
from sgdad.optimizer.base import build_optimizer


def main(argv=None):
    parser = argparse.ArgumentParser(description='Script to train a model')
    parser.add_argument('--config', help='Path to yaml configuration file for the trial')
    parser.add_argument('--model-seed', help='Seed for model\'s initialization')
    parser.add_argument('--sampler-seed', help='Seed for data sampling order')

    args = parser.parse_args(argv)

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    device = torch.device("cuda")

    seed(args.data_sampler_seed, use_cuda)

    train_loader = build_dataset(**config['data'])['train']

    # Note: model is not loaded here for resumed trials
    seed(args.model_init_seed, use_cuda)
    model = build_model(**config['model'])

    optimizer = build_optimizer(model=model, **config['optimizer'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_save_model(engine):
        save_model(model, 'model', engine.state.epoch)

    @trainer.on(Events.STARTED)
    def trainer_load_model(engine):
        metadata = load_model(model, 'model')
        if metadata:
            engine.state.epoch = metadata['epoch']
        else:
            save_model(model, 'model', epoch=0)

    trainer.run(train_loader, max_epochs=args.epochs)


def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
