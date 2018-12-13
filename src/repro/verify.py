import sys
import time

from visdom import Visdom
from kleio.core.io.trial_builder import TrialBuilder
from kleio.core.evc.trial_node import TrialNode
from kleio.core.trial.base import Trial, TrialView


ID_LENGTH = 7

database = TrialBuilder().build_database({})

viz = Visdom(env=";".join(sorted(sys.argv[1:])))
startup_sec = 1
while not viz.check_connection() and startup_sec > 0:
    time.sleep(0.1)
    startup_sec -= 0.1
assert viz.check_connection(), 'No connection could be formed quickly'


analysis = "synthetic"
dataset = sys.argv[1]
model = sys.argv[2]
version = sys.argv[3]

trials = {}

levels = {}

tags = [analysis, dataset, model, version]

def update_trials():
    FIELDS = ['_id', 'commandline', 'configuration', 'version', 'refers', 'host', 'tags']
    trials_found = database.read(
        'trials.reports',
        {'tags': {'$all': tags},
         'registry.status': {'$eq': 'completed'}},
        dict(zip(FIELDS, [1] * 100)))
    print("{} trials found".format(len(trials_found)))
    for trial in trials_found:
        short_id = trial['_id'][:ID_LENGTH]
        if short_id in trials:
            assert trial['_id'] == trials[short_id]['trial'].id
            continue

        trial_id = trial.pop('_id')
        trial_tags = trial.pop('tags')
        if sorted(trial_tags) != sorted(tags):
            print(trial_tags)
            continue

        trial = TrialNode(trial_id, trial=TrialView(Trial(**trial)))
        trial.update()
        # trial = Trial(**trial)
        # print(trial)
        trials[short_id] = {'trial': trial}
        try:
            has_update = "update" in trial.configurations
        except Exception as e:
            print(str(e))
            print(trial.parent.configuration)
            import pdb
            pdb.set_trace()
            raise
        if has_update:
            level = float(trial.configurations['update'].split('=')[-1])
        else:
            level = float(trial.configurations['updates'].split('=')[-1])
        # except:
        #     import pdb
        #     pdb.set_trace()
        if level in levels:
            print("Two trials have the same shuffling level {} and {}".format(
                levels[level].short_id, trial.short_id))
            # import pdb
            # pdb.set_trace()

        print(trial.short_id)
        print(trial.statistics)

        if len(trial.statistics.epoch.keys()) > 0:
            print("ok")
            levels[level] = trial

        # if not trials[short_id]['trial'].statistics.history:
        #     print("No history", trial['_id'])
        # if trials[short_id]['trial'].statistics.history:
        #     break


update_trials()


for level in sorted(levels.keys()):
    trial = levels[level]
    epoch_stats = trial.statistics.epoch
    X = []
    train_loss = []
    valid_error_rates = []
    win = trial.short_id + "_train_curve"
    for epoch in epoch_stats.keys():
        X.append(epoch)
        valid_error_rates.append(epoch_stats[epoch].valid.to_dict()['error_rate'])
        train_loss.append(epoch_stats[epoch].train.to_dict()['loss'])

    viz.line(
        X=X,
        Y=valid_error_rates,
        win="valid_curves",
        update='replace' if viz.win_exists('valid_curves') else None,
        name="level-{}".format(level),
        opts=dict(title="Valid ER", showlegend=True))

    viz.line(
        X=X,
        Y=train_loss,
        win="train_curves",
        update='replace' if viz.win_exists('train_curves') else None,
        name="level-{}".format(level),
        opts=dict(title="Train loss", showlegend=True))


# specify a (dataset, model) pair
# done
# Load the trials
# done

# print learning curve for each wrapper shuffling level
# ok

# Print curves for each pair (metric, shuffling level)

win = "participation_ratio"
for wrapper_name in [dataset, 'shuffling10', 'sphericalcow']:
    for set_name in ['train', 'valid', 'test']:
        X = []
        sets = []
        for level, trial in sorted(levels.items()):
            keys = list(trial.statistics.epoch[300].tags.keys())
            if not keys:
                continue
            print(keys)
            epoch_stats = trial.statistics.epoch[300]
            # import pdb
            # pdb.set_trace()
            print(epoch_stats.tags[keys[0]].to_dict().keys())
            pr = epoch_stats.tags[keys[0]].to_dict()[wrapper_name][set_name]['function']['participation_ratio']
            X.append(level)
            sets.append(pr)

        if not X:
            continue

        print(X)
        print(sets)
        viz.line(
            X=X,
            Y=sets,
            win=win,
            name="{}-{}".format(wrapper_name, set_name),
            update='replace' if viz.win_exists(win) else None,
            opts=dict(title="PR-fct".format(level), showlegend=True))

X = []
sets = []
for level, trial in sorted(levels.items()):
    print(keys)
    epoch_stats = trial.statistics.epoch[300]
    # import pdb
    # pdb.set_trace()
    print(epoch_stats.to_dict().keys())
    error_rate = epoch_stats.to_dict()['valid']['error_rate']
    X.append(level)
    sets.append(error_rate)

print(X)
print(sets)
viz.line(
    X=X,
    Y=sets,
    win="error_rate",
    update='replace' if viz.win_exists('error_rate') else None,
    opts=dict(title="Valid error rate".format(level)))
