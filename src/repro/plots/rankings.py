import json
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import os
import sys

from matplotlib.colors import to_rgb


WIDTH = 3.487
HEIGHT = WIDTH / 1.618

# Prepare matplotlib
plt.rcParams.update({'font.size': 8})
plt.close('all')
plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)


def plot_all(in_path):
    np.random.seed(1)
    # Get data
    in_path = 'new.json'
    loaded = json.load(open(in_path))
    models = list(sorted(loaded['models'], reverse=True))
    datasets = loaded['datasets']
    data = loaded['data']
    colors = loaded['colors']

    fig, axes = plt.subplots(nrows=1, ncols=1)  # , sharex=True,
                             # gridspec_kw={'height_ratios': [16, 1]})

    # Ordering Histogram
    for i, dataset in enumerate(datasets):
        padding = 1
        M, N = 1000, 10
        img = np.zeros((M, N, 3))
        avg = -np.ones((1, N, 3))
        
        sampled_models = -np.ones((M, N), dtype=np.int64)
        for m in range(M):
            scores = [np.random.choice(data[dataset][mod]) for mod in models]
            indices = np.argsort(scores)
            sampled_models[m] = indices
    
        average_positions = -np.ones(N)
        for j, model in enumerate(models):
            indices = np.where(sampled_models == j)[1]
            average_positions[j] = indices.mean()

        for k, [j, model] in enumerate(sorted(enumerate(models), key=lambda k: average_positions[k[0]])):
            heights, positions = np.histogram(np.where(sampled_models == j)[1], range=(0, N), bins=N, density=True)
            axes.bar(x=positions[:-1], height=heights, bottom=-((i * (N + padding)) + k),
                    color=colors[model], width=1)
            # axes.vlines(average_positions[j], -(i * (N + padding) - 1), -((i + 1) * (N + padding) -
            #     padding),
            #         color=colors[model])
            # axes.vlines(indices.mean(), 0, -(len(datasets) * (N + padding) - padding),
            #         color=colors[model], alpha=0.1)

        axes.text(-3, -((i * (N + padding)) + (j + padding) / 3), dataset, fontsize=12)

        # if i < len(datasets) - 1:
        #     axes.hlines(-((i * (N + padding)) + j + padding / 2), -1.5, N, linewidths=1,
        #             linestyles=(0, (5, 10)), alpha=0.2)

        average_model = np.argsort(average_positions)
        print(positions)
        print(heights)
        # axes.eventplot(np.where(sampled_models == i)[1], colors=colors1, lineoffsets=lineoffsets1,
        #             linelengths=linelengths1)
        print(average_model)
    
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)
        axes.tick_params(axis='both', which='both', bottom=False, top=False,
                         left=False, right=False, labeltop=False,
                         labelbottom=True, labelleft=False, labelright=False)

        axes.set_xticks(range(0, N))
        axes.set_xticklabels(range(1, N + 1))
        axes.set_xlabel('Rankings', fontsize=10)
    
        fig.set_size_inches(HEIGHT, WIDTH)
        fig.tight_layout()
        

    print(i, j)

    plt.show()
    # if not os.path.exists('out'):
    #     os.mkdir('out')
    # plt.savefig(os.path.join('out', out))


plot_all('new.json')
