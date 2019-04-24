"""
    Analize a report
"""

import pandas as pd
import copy
import json


data = json.load(open('dat.json', 'r'))

normalized = []

for configurator, run in data.items():
    for problem_id, trials in run.items():

        row_base = [configurator]
        row_base.extend(problem_id.split(','))

        for trial_id, trial in enumerate(trials):

            row = copy.deepcopy(row_base)

            # we want params to be last
            row.append(trial_id)
            row.append(trial['objective'])
            row.append(trial['hpo_time'])
            row.append(trial['exec_time'])

            for param in trial['params']:
                row.append(param)

            normalized.append(row)


columns = [
    'configurator',
    'task',
    'fid',
    'dim',
    'instance',
    'scenario',
    'warm',
    'workers',
    'previous',
    'trial_id',
    'objectives',
    'hpo_time',
    'exec_time'
    # ... params
]
df = pd.DataFrame(normalized)


r, c = df.shape
for i in range(0, c - len(columns)):
    columns.append(f'param_{i}')

df.columns = columns
df.to_csv('dat.csv', index=False)

