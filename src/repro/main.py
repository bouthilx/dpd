import os
import time
import sys
import logging
import json
from orion.core.io.space_builder import Space, DimensionBuilder

from hpo.manager import HPOManager
from hpo.dispatcher.asha import ASHA
from hpo.dispatcher.median_stopping_rule import MedianStoppingRule
from hpo.dispatcher.dpf import DPF
from hpo.dispatcher.stub import Stub
from vgg_example import main
from utils.flatten import flatten

logging.basicConfig(level=logging.DEBUG)


def build_space():
    space = Space()
    dimension_builder = DimensionBuilder()
    full_space_config = {'lr': 'loguniform(1.0e-5, 1)',
                         'dropout': 'uniform(0.0, 0.6)',
                         'momentum': 'uniform(0.0, 0.99)',
                         'weight_decay': 'loguniform(1.0e-8, 1.0e-3)'}

    for name, prior in flatten(full_space_config).items():
        if not prior:
            continue
        try:
            space[name] = dimension_builder.build(name, prior)
        except TypeError as e:
            print(str(e))
            print('Ignoring key {} with prior {}'.format(name, prior))
    return space

timer = time.time()
algo = sys.argv[1]
out_file = 'dispatcher=' + algo + '.json'
max_iterations = 4072
print(out_file)
print(os.getcwd())
space = build_space() 
if algo == 'asha':
    brackets = int(sys.argv[2])
    out_file = 'dispatcher=' + sys.argv[1] + ',brackets=' + sys.argv[2] + '.json'
    dispatcher = ASHA(space, configurator_config=dict(name='random_search', seed=10),
                      max_epochs=120, grace_period=1, reduction_factor=4, brackets=brackets,
                      seed=0)
elif algo == 'msr':
    dispatcher = MedianStoppingRule(space, dict(name='random_search', seed=10),
                                    seed=0, grace_period=10, min_samples_required=3)
elif algo == 'dpf':
    step_ratio = float(sys.argv[2])
    out_file = 'dispatcher=' + algo + ',step_ratio=' + sys.argv[2] + '.json'
    dispatcher = DPF(space, dict(name='random_search', seed=10),
                     seed=0, steps_ratio=step_ratio,
                     asynchronicity=0.5, final_population=3, max_epochs=120)
else:
    if algo == 'bayesopt':
        config = {'name': 'bayesopt',
                  'strategy': 'cl_min',
                  'n_initial_points': 32,
                  'acq_func': 'gp_hedge',
                  'alpha': 1.0e-10,
                  'n_restarts_optimizer': 0,
                  'normalize_y': False,
                  'noise': None}
    elif algo == 'tpe':
        config = {'name': 'tpe',
                  'consider_prior': True,
                  'prior_weight': 1.0,
                  'consider_magic_clip': True,
                  'consider_endpoints': False,
                  'n_startup_trials': 32,
                  'n_ei_candidates': 24}
    elif algo == 'random':
        config = {'name': 'random_search'}
    else:
        raise NotImplementedError
    config.update({'seed': 10})
    dispatcher = Stub(space, configurator_config=config, seed=0)

manager = HPOManager(None, dispatcher, task=main, max_iterations=max_iterations,
                     gpus=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'] * 8)

manager.run()

data = json.dumps([trial.to_dict() for trial in manager.trials], indent=4)
json_file = open(out_file, 'w')
json_file.write(data)
json_file.write('\n')
json_file.close()

print(time.time() - timer)
