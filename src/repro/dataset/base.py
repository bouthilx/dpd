import os

from repro.utils.factory import fetch_factories
from repro.dataset.wrapper.base import build_wrapper


factories = fetch_factories('repro.dataset', __file__)


def set_data_path(config):
    if "REPRO_DATA_PATH" not in os.environ:
        print('WARNING: Environment variable REPRO_DATA_PATH is not set. '
              'Data will be downloaded in {}'.format(os.getcwd()))

    config['data_path'] = os.environ.get('REPRO_DATA_PATH', os.getcwd())


def build_dataset(name=None, **kwargs):
    set_data_path(kwargs)
    # TODO: Put back in after rebuttal
    # kwargs['num_workers'] = 0
    wrapper = kwargs.pop('wrapper', None)
    data = factories[name](**kwargs)

    if 'input_size' not in data or 'num_classes' not in data:
        # Get one batch
        data['input_size'] = 0
        data['num_classes'] = 0

        # Run over many mini-batches to make sure all classes are seen
        for i, (source, target) in enumerate(data['train']):
            data['input_size'] = list(source.size()[1:])
            data['num_classes'] = max(data['num_classes'], target.max().item() + 1)

            if i > 20:
                break

    if wrapper:
        data = build_wrapper(data, **wrapper)

    return data
