from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.hpo.trial', __file__)


def build_trial(name='builtin', **kwargs):
    return factories[name](**kwargs)
