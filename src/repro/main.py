import logging
import json
from orion.core.io.space_builder import Space, DimensionBuilder

from hpo.manager import HPOManager
from hpo.dispatcher.asha import ASHA
from vgg_example import main
from utils.flatten import flatten

logging.basicConfig(level=logging.DEBUG)

def build_space():
    space = Space()
    dimension_builder = DimensionBuilder()
    full_space_config = {'lr': 'loguniform(1.0e-4, 10)'}

    for name, prior in flatten(full_space_config).items():
        if not prior:
            continue
        try:
            space[name] = dimension_builder.build(name, prior)
        except TypeError as e:
            print(str(e))
            print('Ignoring key {} with prior {}'.format(name, prior))
    return space

resource_manager = None
max_trials = 512 

space = build_space() 
dispatcher = ASHA(space, configurator_config=dict(name='random_search', max_trials=max_trials, seed=10),
                  fidelities=[15, 30, 60, 120], reduction_factor=4, max_resource=8,
                  max_trials=max_trials, seed=0)

manager = HPOManager(resource_manager, dispatcher, task=main, max_trials=max_trials,
                     gpus=['cuda:0'] * 8 + ['cuda:1'] * 8 + ['cuda:2'] * 8 + ['cuda:3'] * 8)

manager.run()

data = json.dumps([trial.to_dict() for trial in manager.trials], indent=4)
json_file = open('test.json', 'w')
json_file.write(data)
json_file.write('\n')
json_file.close()
