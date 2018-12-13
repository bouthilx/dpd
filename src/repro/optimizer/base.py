from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.optimizer', __file__)


def build_optimizer(name, **kwargs):
    return factories[name](**kwargs)
