from sgdad.utils.factory import fetch_factories


factories = fetch_factories('sgdad.dataset.wrapper', __file__)


def build_wrapper(data, **kwargs):
    return factories[name](**kwargs)
