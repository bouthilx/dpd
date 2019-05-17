import os
import time
import sys
import logging
import json
import numpy as np

from orion.core.io.space_builder import Space, DimensionBuilder

from hpo.manager import HPOManager
from hpo.dispatcher.asha import ASHA
from hpo.dispatcher.median_stopping_rule import MedianStoppingRule
from hpo.dispatcher.dpf import DPF
from hpo.dispatcher.stub import Stub
from vgg_example import main
from utils.flatten import flatten

logging.basicConfig(level=logging.DEBUG)


def build_space(full_space_config):
    space = Space()
    dimension_builder = DimensionBuilder()

    for name, prior in flatten(full_space_config).items():
        if not prior:
            continue
        try:
            space[name] = dimension_builder.build(name, prior)
        except TypeError as e:
            print(str(e))
            print('Ignoring key {} with prior {}'.format(name, prior))
    return space


def curves(a, b, callback=None, **kwargs):

    def forget_curve(t, a=1.84, b=1.25, c=1.84):
        o = a / (np.log(t) ** b + c)
        return (o <= 1) * o + (o > 1)

    def gated(t, a, b):
        return forget_curve(t, a=max(a, b), b=a, c=a)

    T = 100
    objectives = []
    for t in range(1, T + 1):
        y = gated(t, a, b)
        time.sleep(1)
        if callback is not None:
            callback(step=t, objective=y, finished=(t) >= T)
        objectives.append({'epoch': t, 'objective': y})


def main_asha(callback=None, id=None, device=None, data_path=None, xp_path=None, **kwargs):
    max_trials = 512 
    space = build_space({'a': 'uniform(0.1, 2.0)', 'b': 'uniform(0, 10.0)'})
    configurator_config=dict(name='random_search', max_trials=max_trials, seed=10)
    dispatcher = ASHA(space, configurator_config=configurator_config,
                      max_trials=max_trials, seed=0, max_epochs=100, **kwargs)
    manager = HPOManager(None, dispatcher, task=curves, max_trials=max_trials,
                         gpus=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'] * 8)
    manager.run()
    data = [trial.to_dict() for trial in manager.trials]
    best_objective = 1000
    for d in data:
        best_objective = min([r['objective'] for r in d['results']] + [best_objective])
    if callback is not None:
        callback(step=0, objective=best_objective, finished=True)


def main_dpf(callback=None, id=None, device=None, data_path=None, xp_path=None, **kwargs):
    max_trials = 512
    space = build_space({'a': 'uniform(0.1, 2.0)', 'b': 'uniform(0, 10.0)'})
    configurator_config=dict(name='random_search', max_trials=max_trials, seed=10)
    dispatcher = DPF(space, configurator_config=configurator_config,
                     max_trials=max_trials, seed=0, final_population=10,
                     max_epochs=100, **kwargs)
    manager = HPOManager(None, dispatcher, task=curves, max_trials=512,
                         gpus=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'] * 8)
    manager.run()
    data = [trial.to_dict() for trial in manager.trials]
    best_objective = 1000
    for d in data:
        best_objective = min([r['objective'] for r in d['results']] + [best_objective])
    if callback is not None:
        callback(step=0, objective=best_objective, finished=True)


meta_max_trials = 10 
meta_config = {'name': 'random_search', 'max_trials': meta_max_trials, 'seed': 10}
if sys.argv[1] == 'asha':
    main = main_asha
    out_file = 'task=curves,dispatcher=asha.json'  
    full_space_config = {'grace_period': 'loguniform(1, 50, discrete=True)',
                         'reduction_factor': 'uniform(2, 5, discrete=True)',
                         'brackets': 'uniform(1, 4, discrete=True)'}
elif sys.argv[1] == 'dpf':
    main = main_dpf
    out_file = 'task=curves,dispatcher=dpf.json'  
    full_space_config = {'steps_ratio': 'uniform(0, 1)',
                         'asynchronicity': 'uniform(0, 1)',
                         'window_size': 'uniform(1, 11, discrete=True)'}
else:
    raise NotImplementedError

meta_space = build_space(full_space_config)
meta_dispatcher = Stub(meta_space, configurator_config=meta_config, max_trials=meta_max_trials, seed=0)
meta_manager = HPOManager(None, meta_dispatcher, task=main, max_trials=meta_max_trials,
                          gpus=['toto'])
meta_manager.run()
meta_data = json.dumps([trial.to_dict() for trial in meta_manager.trials], indent=4)
json_file = open(out_file, 'w')
json_file.write(meta_data)
json_file.write('\n')
json_file.close()
