from sgdad.utils.factory import fetch_factories
from sgdad.dataset.wrapper.base import build_wrapper


factories = fetch_factories('sgdad.dataset', __file__)


def build_dataset(name, **kwargs):
    wrapper = kwargs.pop('wrapper', None)
    data = factories[name](**kwargs)

    if wrapper:
        return builder_wrapper(data, **wrapper)

    return data
