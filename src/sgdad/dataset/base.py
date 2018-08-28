import os

from sgdad.utils.factory import fetch_factories
from sgdad.dataset.wrapper.base import build_wrapper


factories = fetch_factories('sgdad.dataset', __file__)


def set_data_path(config):
    if "SGD_SPACE_DATA_PATH" not in os.environ:
        raise RuntimeError("Environment variable SGD_SPACE_DATA_PATH is not set")

    config['data_path'] = os.environ['SGD_SPACE_DATA_PATH']


def build_dataset(name=None, **kwargs):
    set_data_path(kwargs)

    wrapper = kwargs.pop('wrapper', None)
    data = factories[name](**kwargs)

    # Get one batch
    source, target = next(iter(data['train']))
    data['input_size'] = list(source.size()[1:])
    data['num_classes'] = target.max().item() + 1

    if wrapper:
        data = build_wrapper(data, **wrapper)

    return data
