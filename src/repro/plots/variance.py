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

# Get data
def plot(in_path):
    np.random.seed(1)
    loaded = json.load(open(in_path))
    models = list(sorted(loaded['models'], reverse=True))
    datasets = loaded['datasets']
    data = loaded['data']
    colors = loaded['colors']
    
    for i, dataset in enumerate(datasets):
        fig, axe = plt.subplots(nrows=1, ncols=1, sharex=True)
        M, N = 16, 8
        img = np.zeros((M, N, 3))
        avg = -np.ones((1, N, 3))
        
        sampled_models = -np.ones((M, N), dtype=np.int64)
        for m in range(M):
            scores = [np.random.choice(data[dataset][model]) for model in models]
            indices = np.argsort(scores)
            sampled_models[m] = indices
    
        average_positions = -np.ones(N)
        variance_positions = -np.ones(N)
        for i, model in enumerate(models):
            indices = np.where(sampled_models == i)[1]
            average_positions[i] = indices.mean()
            variance_positions[i] = indices.var()
        average_model = np.argsort(average_positions)
    
        cs = []
        for n in range(N):
            c = matplotlib.colors.to_rgb(colors[models[average_model[n]]])
            avg[0, n] = c
            cs.append(c)
       
        variance_positions = variance_positions[average_model]
        variance_positions += 0.1 * np.max(variance_positions)  # to see all columns
        axe.bar(range(len(variance_positions)), variance_positions, width=1, color=cs)
        axe.set_ylabel('Variance')
        axe.set_xlabel('Ranking')
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.spines['bottom'].set_visible(False)
        axe.spines['left'].set_visible(False)
        axe.tick_params(axis='both', which='both', bottom=False, top=False,
                        left=False, right=False, labeltop=False,
                        labelbottom=False, labelleft=False, labelright=False)
    
        fig.set_size_inches(WIDTH, HEIGHT)
        out = dataset + '_' + os.path.splitext(in_path)[0] + '_var.pdf'
        #plt.show()
        if not os.path.exists('out'):
            os.mkdir('out')
        plt.savefig(os.path.join('out', out))


def plot_all(in_path):                
    np.random.seed(1)
    loaded = json.load(open(in_path))
    models = list(sorted(loaded['models'], reverse=True))
    datasets = loaded['datasets']
    data = loaded['data']
    colors = loaded['colors']

    M, N = 16, 8
    img = np.zeros((M*len(datasets), N, 3))
    sampled_models = -np.ones((M*len(datasets), N), dtype=np.int64)
    for i, dataset in enumerate(datasets):
        for m in range(M):
            scores = [np.random.choice(data[dataset][model]) for model in models]
            indices = np.argsort(scores)
            sampled_models[m + i*M] = indices
    
    average_positions = -np.ones(N)
    variance_positions = -np.ones(N)
    for i, model in enumerate(models):
        indices = np.where(sampled_models == i)[1]
        average_positions[i] = indices.mean()
        variance_positions[i] = indices.var()
    average_model = np.argsort(average_positions)
    variance_positions = variance_positions[average_model]
    
    cs = []
    for n in range(N):
        c = matplotlib.colors.to_rgb(colors[models[average_model[n]]])
        cs.append(c)
    
    fig, axe = plt.subplots(nrows=1, ncols=1, sharex=True)
    variance_positions += 0.1 * np.max(variance_positions)  # to see all bars
    axe.bar(range(len(variance_positions)), variance_positions, width=1,
            color=cs)
    axe.set_ylabel('Variance')
    axe.set_xlabel('Ranking')
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)
    axe.spines['bottom'].set_visible(False)
    axe.spines['left'].set_visible(False)
    axe.tick_params(axis='both', which='both', bottom=False, top=False,
                    left=False, right=False, labeltop=False,
                    labelbottom=False, labelleft=False, labelright=False)
    
    fig.set_size_inches(WIDTH, HEIGHT)
    out = os.path.splitext(in_path)[0] + '_var.pdf'
    #plt.show()
    if not os.path.exists('out'):
        os.mkdir('out')
    plt.savefig(os.path.join('out', out))

plot('seed.json')
plot('hpo.json')
plot_all('seed.json')
plot_all('hpo.json')
