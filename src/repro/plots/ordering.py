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
    in_path = 'hpo.json'
    loaded = json.load(open(in_path))
    models = list(sorted(loaded['models'], reverse=True))
    datasets = loaded['datasets']
    data = loaded['data']
    colors = loaded['colors']
    
    # Ordering Histogram
    for i, dataset in enumerate(datasets):
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                                 gridspec_kw={'height_ratios': [16, 1]})
        M, N = 16, 8
        img = np.zeros((M, N, 3))
        avg = -np.ones((1, N, 3))
        
        sampled_models = -np.ones((M, N), dtype=np.int64)
        for m in range(M):
            scores = [np.random.choice(data[dataset][mod]) for mod in models]
            indices = np.argsort(scores)
            sampled_models[m] = indices
    
        average_positions = -np.ones(N)
        for i, model in enumerate(models):
            indices = np.where(sampled_models == i)[1]
            average_positions[i] = indices.mean()
        average_model = np.argsort(average_positions)
    
        for n in range(N):
            avg[0, n] = to_rgb(colors[models[average_model[n]]])
            for m in range(M):
                img[m, n] = to_rgb(colors[models[sampled_models[m, n]]])
        
        axes[0].imshow(img, interpolation='none', aspect='equal')
        axes[1].imshow(avg, interpolation='none', aspect='equal')
        axes[0].set_ylabel('Samples')
        axes[1].set_ylabel('Avg')
        axes[1].set_xlabel('Ranking')
    
        for axe in axes:
            axe.spines['top'].set_visible(False)
            axe.spines['right'].set_visible(False)
            axe.spines['bottom'].set_visible(False)
            axe.spines['left'].set_visible(False)
            axe.tick_params(axis='both', which='both', bottom=False, top=False,
                            left=False, right=False, labeltop=False,
                            labelbottom=False, labelleft=False, labelright=False)
    
        fig.set_size_inches(HEIGHT, WIDTH)
        fig.tight_layout()
        out = dataset + '_' + os.path.splitext(in_path)[0] + '_ordering.pdf'
        #plt.show()
        if not os.path.exists('out'):
            os.mkdir('out')
        plt.savefig(os.path.join('out', out))


plot_all('seed.json')
plot_all('hpo.json')
