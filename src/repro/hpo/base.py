from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.hpo', __file__)


def build_hpo(space, name=None, **kwargs):
    return factories[name](space, **kwargs)
