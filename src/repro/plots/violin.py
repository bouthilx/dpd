import json
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import os
import sys


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
    # Get data
    loaded = json.load(open(in_path))
    models = list(sorted(loaded['models'], reverse=True))
    datasets = loaded['datasets']
    data = loaded['data']
    colors = loaded['colors']
    
    # Seed Histograms
    for i, dataset in enumerate(datasets):
        fig, axe = plt.subplots(1, 1)
        for i, m in enumerate(models):
            x = data[dataset][m]
            parts = axe.violinplot(x, [i+1], points=8, showmeans=False,
                                   showextrema=False, showmedians=False,
                                   vert=False)
            for p in parts['bodies']:
                p.set_facecolor(colors[m])
                p.set_alpha(1)
            axe.scatter(x, [i+1]*len(x), s=1, color='k')
        axe.set_xlabel('Performances')
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.spines['bottom'].set_visible(False)
        axe.spines['left'].set_visible(False)
        axe.yaxis.tick_right()
        axe.set_yticks(range(1, len(models) + 1))
        axe.set_yticklabels(models)
        if dataset == 'cifar10':
            axe.set_xlim(0.1, 0.25)
        elif dataset == 'cifar100':
            axe.set_xlim(0.35, 0.55)
        axe.tick_params(axis='x', which='both', bottom=False, top=False,
                        left=False, right=False, labeltop=False,
                        labelbottom=False, labelleft=False, labelright=False)
        axe.tick_params(axis='y', which='both', bottom=False, top=False,
                        left=False, right=False, labeltop=False,
                        labelbottom=False, labelleft=False, labelright=True)
        fig.set_size_inches(WIDTH, HEIGHT)
        fig.tight_layout()
        out = dataset + '_' + os.path.splitext(in_path)[0] + '_violin.pdf'
        #plt.show()
        if not os.path.exists('out'):
            os.mkdir('out')
        plt.savefig(os.path.join('out', out))


plot_all('seed.json')
plot_all('hpo.json')
