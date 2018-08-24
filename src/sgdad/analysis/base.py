from sgdad.utils.factory import fetch_factories


factories = fetch_factories('sgdad.analysis', __file__)


def build_analysis(name, **kwargs):
    return factories[name](**kwargs)
