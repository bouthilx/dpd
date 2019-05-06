from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.hpo.resource', __file__)


def build_resource_manager(name='builtin', **kwargs):
    return factories[name](**kwargs)
