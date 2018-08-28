from glob import glob
import os


def fetch_factories(base_module, base_file_name):
    factories = {}
    module_path = os.path.dirname(os.path.abspath(base_file_name))
    for module_path in glob(os.path.join(module_path, '[A-Za-z]*.py')):
        module_file = module_path.split(os.sep)[-1]

        if module_file == base_file_name:
            continue

        module_name = module_file.split(".py")[0]
        module = __import__(".".join([base_module, module_name]), fromlist=[''])

        if hasattr(module, 'build'):
            factories[module_name] = module.build

    return factories
