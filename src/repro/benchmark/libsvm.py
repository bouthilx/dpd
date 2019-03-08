import itertools
import logging
import re
import os
import urllib.request

import numpy

try:
    from lxml import etree
    import lxml.html
    from sklearn.datasets import load_svmlight_file
    from tqdm import tqdm
    import scipy.sparse
except ImportError:
    etree = None


try:
    import mahler.client as mahler
except ImportError:
    mahler = None


logger = logging.getLogger(__name__)


file_name_regex = "^{}((\.|_)scale)?(\.t)?(\.bz2)?$"
scale_regex = "^{}((\.|_)scale)(\.t)?(\.bz2)?$"


def download_data(name):
    # Try in binary, otherwise try in multi-class

    def _download_data(name, url):
        fp = urllib.request.urlopen(url)
        mybytes = fp.read()
        mystr = mybytes.decode("utf8")
        fp.close()
        root = lxml.html.fromstring(mystr)

        filenames = []
        for tr in root.getchildren()[1].getchildren()[1].getchildren()[3:-1]:
            filename = tr.getchildren()[1].getchildren()[0].text
            if re.match(file_name_regex.format(name), filename):
                filenames.append(filename)

        name_scale_regex = re.compile(scale_regex.format(name))
        if any(not name_scale_regex.match(filename) for filename in filenames):
            filenames = [filename for filename in filenames if not name_scale_regex.match(filename)]
        print(name, filenames)

        return filenames

    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    filenames = _download_data(name, url)

    if not filenames:
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
        filenames = _download_data(name, url)

    data = []
    for filename in filenames:
        # TODO: Set path where to download file
        download(filename, url + filename, filename)
        data.append(load_svmlight_file(filename))

    import pdb
    pdb.set_trace()

    # TODO:
    # Get shape of each
    # Make zero template, and push csr into it
    # Stack both
    # Turn the stacked matrix into csr matrix
    # Return

    x = scipy.sparse.vstack([subset[0] for subset in data])
    y = numpy.stack([subset[1] for subset in data])

    return x, y


def download(name, url, data_path):
    if os.path.exists(data_path):
        print("Zip file already downloaded")
        return

    # download 
    u = urllib.request.urlopen(url)
    with open(data_path, 'wb') as f:
        file_size = int(dict(u.getheaders())['Content-Length']) / (10.0**6)
        print("Downloading: {} ({}MB)".format(data_path, file_size))

        block_sz = 8192
        pbar = tqdm(total=file_size, desc=name)
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            f.write(buffer)
            pbar.update(len(buffer) / (10.0 ** 6))

        pbar.close()


def load_dataset_names():
    fp = urllib.request.urlopen("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/")
    mybytes = fp.read()

    mystr = mybytes.decode("utf8")
    fp.close()

    root = etree.fromstring(mystr)

    headers = [e.text for e in
               root.getchildren()[1].getchildren()[8].getchildren()[0].getchildren()]

    name_index = headers.index('name')
    training_size_index = headers.index('training size')
    testing_size_index = headers.index('testing size')
    feature_size_index = headers.index('feature')
    type_index = headers.index('type')

    dataset_names = []

    for tr in root.getchildren()[1].getchildren()[8].getchildren()[1:]:
        values = [e.getchildren()[0].text if e.getchildren() else e.text for e in tr.getchildren()]

        if values[type_index] not in ['classification']:
            continue

        name = values[name_index]
        training_size = int(values[training_size_index].replace(",", ""))
        testing_size = int(values[testing_size_index].replace(",", "")
                           if values[testing_size_index] else 0)
        feature_size = int(values[feature_size_index].replace(",", ""))
        if training_size + testing_size > 40000:
            continue
        elif feature_size > 300:
            continue

        dataset_names.append(name)

        download_data(name)

    return dataset_names


class LIBSVMBenchmark:

    name = 'libsvm'

    def __init__(self, datasets=None, dataset_folds=None, processings=None, scenarios=None,
                 previous_tags=None, warm_start=0):
        self.verify(datasets, dataset_folds, processings, scenarios)
        self._datasets = datasets
        self._dataset_folds = dataset_folds
        self._processings = processings
        self._scenarios = scenarios
        self._previous_tags = previous_tags
        self._warm_start = warm_start

    def verify(self, datasets, dataset_folds, processings, scenarios):
        # TODO
        pass

    def add_subparser(self, subparsers):
        benchmark_parser = subparsers.add_parser('libsvm')

        benchmark_parser.add_argument('--datasets', choices=self.datasets, nargs='*', type=str)
        benchmark_parser.add_argument('--dataset-folds', choices=self.dataset_folds, nargs='*',
                                      type=int)
        benchmark_parser.add_argument('--processings', choices=self.processings, type=str, nargs='*')
        benchmark_parser.add_argument('--scenarios', choices=self.scenarios, type=str, nargs='*')
        benchmark_parser.add_argument('--warm-start', type=int, default=50)
        benchmark_parser.add_argument('--max-trials', type=int, default=50)

        return benchmark_parser

    @property
    def datasets(self):
        if getattr(self, '_datasets', None):
            return self._datasets

        return load_dataset_names()

    @property
    def dataset_folds(self):
        if getattr(self, '_dataset_folds', None):
            return self._dataset_folds

        return list(range(1, 6))

    @property
    def processings(self):
        if getattr(self, '_datasets', None):
            return self._datasets

        return ['none', 'pca', 'zca']

    @property
    def scenarios(self):
        if getattr(self, '_scenarios', None):
            return self._scenarios

        # TODO: Define how to set 2.1, 2.2 and 2.3.
        return ['0.0',  # No diff
                ]  # H-P is scaled

    @property
    def problems(self):
        configs = itertools.product(self.datasets, self.dataset_folds, self.processings,
                                    self.scenarios)

        for config in configs:
            yield Problem(*(config + (self._previous_tags, self._warm_start)))


class Problem:
    def __init__(self, problem_id, dimension, instance_id, scenario, previous_tags, warm_start):
        self.id = problem_id
        self.dimension = dimension
        self.instance_id = instance_id
        self.scenario = scenario
        self.previous_tags = previous_tags
        self.warm_start = warm_start

    @property
    def tags(self):
        tags = ['libsvm',
                'd-{}'.format(self.dataset_name),
                'df-{}'.format(self.dataset_fold),
                'p-{}'.format(self.processing),
                's-{}'.format(self.scenario),
                'm-{}'.format(self.warm_start)]
        if self.previous_tags:
            tags += ['pv-{}'.format(tag) for tag in self.previous_tags]
        else:
            tags += ['pv-none']

        return tags

    # if 0., nothing
    # if 2.1, drop one?
    # if 2.2 add one?
    # if 2.2 rm one?
    # if 2.4.a (lower, mid)
    # if 2.4.b (mid, upper)

        # config = dict(
        #     prior='uniform', lower_bound=problem.lower_bounds[dim],
        #     upper_bound=problem.upper_bounds[dim])
        # config.update(space_config.get(name, {}))

    @property
    def space_config(self):
        return {}

    @property
    def config(self):
        return dict(
            problem_id=self.id,
            nb_of_dimensions=self.dimension,
            instance_id=self.instance_id)

    def execute(self, configurator_config):
        # rval = hpo_coco(
        #     self.config,
        #     self.space_config,
        #     configurator_config=configurator_config,
        #     previous_tags=self.previous_tags,
        #     warm_start=self.warm_start)

        # print(self.id, self.dimension, self.instance_id)
        # print(len(rval['objectives']), min(rval['objectives']))
        pass

    def register(self, mahler_client, configurator_config, container, tags):

        # import pdb
        # pdb.set_trace()

        # mahler_client.register(
        #     hpo_coco.delay(
        #         problem_config=self.config,
        #         space_config=self.space_config,
        #         configurator_config=configurator_config,
        #         previous_tags=self.previous_tags,
        #         warm_start=self.warm_start),
        #     container=container, tags=tags)

        # print('Registered', *tags)
        pass

    def visualize(self, results, filename_template):
        # TODO: Add visualize to benchmark as well, where it compares problems together.

        PHASES = ['warmup', 'cold-turkey', 'transfer']

        import matplotlib.pyplot as plt
        ALPHA = 0.1

        colors = {
            'random_search': 'blue',
            'bayesopt': 'red'}

        # Plot warmup
        for configurator_name, configurator_results in results.items():
            for instance_results in configurator_results['warmup']:
                plt.plot(range(len(instance_results)), instance_results,
                         color=colors[configurator_name], alpha=ALPHA)
                plt.plot(range(len(instance_results)),
                         [min(instance_results[:i]) for i in range(1, len(instance_results) + 1)],
                         color=colors[configurator_name], linestyle=':')

        # Plot cold-turkey
        for configurator_name, configurator_results in results.items():
            for instance_results in configurator_results['cold-turkey']:
                x = list(range(self.warm_start, len(instance_results) + self.warm_start))
                plt.plot(x, instance_results,
                         linestyle='--', color=colors[configurator_name], alpha=ALPHA)
                plt.plot(x,
                         [min(instance_results[:i]) for i in range(1, len(instance_results) + 1)],
                         color=colors[configurator_name], linestyle='--')

        # Plot transfer
        for configurator_name, configurator_results in results.items():
            for i, instance_results in enumerate(configurator_results['transfer']):
                x = list(range(self.warm_start, len(instance_results) + self.warm_start))
                plt.plot(x, instance_results,
                         color=colors[configurator_name], alpha=ALPHA)
                plt.plot(x,
                         [min(instance_results[:i]) for i in range(1, len(instance_results) + 1)],
                         color=colors[configurator_name],
                         label=configurator_name if i == 0 else None)

        plt.axvline(x=self.warm_start, linestyle='--')

        plt.title("f{:03d}-d{:03d}-s{}".format(self.id, self.dimension, self.scenario))

        plt.legend()
        file_path = filename_template.format(id=self.id, dimension=self.dimension,
                                             scenario=self.scenario)
        plt.savefig(file_path, dpi=300)
        print("Saved", file_path)
        plt.clf()


def build(datasets=None, dataset_folds=None, processings=None, scenarios=None, previous_tags=None,
          warm_start=None, **kwargs):
    return LIBSVMBenchmark(datasets, dataset_folds, processings, scenarios, previous_tags, warm_start)
