from sgdad.utils.factory import fetch_factories


factories = fetch_factories('sgdad.dataset.wrapper', __file__)


def build_wrapper(data, name=None, **kwargs):
    return factories[name](data, **kwargs)
