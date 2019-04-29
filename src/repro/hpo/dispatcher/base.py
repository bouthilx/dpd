from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.hpo.dispatcher', __file__)


def build_dispatcher(space, name=None, **kwargs):
    return factories[name](space, **kwargs)
