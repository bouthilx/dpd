from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.hpo.configurator', __file__)


def build_configurator(space, name=None, **kwargs):
    return factories[name](space, **kwargs)
