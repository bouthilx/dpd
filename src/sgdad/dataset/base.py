from glob import glob
import os


def fetch_dataset_factories():
    factories = {}
    base_module = 'sgdad.dataset'
    module_path = os.path.dirname(os.path.abspath(__file__))
    for module_path in glob(os.path.join(module_path, '[A-Za-z]*.py')):
        module_file = module_path.split(os.sep)[-1]

        if module_file == __file__:
            continue

        module_name = module_file.split(".py")[0]
        module = __import__(".".join([base_module, module_name]), fromlist=[''])

        if hasattr(module, 'build'):
            factories[module_name] = module.build

    return factories


def build_dataset(name, **kwargs):
    return factories[name](**kwargs)
    

factories = fetch_dataset_factories()
