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

    max_features = max(subset[0].shape[1] for subset in data)

    x = numpy.vstack(
        [numpy.hstack([subset[0].todense(),
                       numpy.zeros((subset[0].shape[0], max_features - subset[0].shape[1]))])
         for subset in data])
    x = scipy.sparse.csr_matrix(x)

    y = numpy.concatenate([subset[1] for subset in data])

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
    return []
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
        benchmark_parser.add_argument('--processings', choices=self.processings, type=str,
                                      nargs='*')
        benchmark_parser.add_argument('--scenarios', choices=self.scenarios, type=str, nargs='*')
        benchmark_parser.add_argument('--warm-start', type=int, default=50)

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

        # PHASES = ['warmup', 'cold-turkey', 'transfer']

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


def execute():

    pass


# TODO: Pass model names from commandline and then automatically build the model configs
#       then register those configs with mahler.
def build_model_config(model_names):
    model_config = {}
    for model_name in model_names:
        if model_name in model_config:
            ith_instance = len(model_config[model_name])
        else:
            model_config[model_name] = {}
            ith_instance = 0

        if ith_instance in SPACES[model_name]:
            space = SPACES[model_name][ith_instance]
        else:
            space = SPACES[model_name][0]

        space = copy.deepcopy(space)
        space['weight'] = 'uniform(0, 1)'
        model_config[model_name][ith_instance] = space

    return model_config


# TODO:
def build_space(model_config):
    space = Space()
    space_config = {}
    for name, prior in flatten(model_config).items():
        if not prior:
            continue
        try:
            space[name] = dimension_builder.build(name, prior)
        except TypeError as e:
            print(str(e))
            print('Ignoring key {} with prior {}'.format(name, prior))

    return space


SPACES = dict(
    KNeighborsClassifier={
        0: dict(
            algorithm='brute',
            n_neighbors='uniform(1, 20, discrete=True)',
            weights='choices(["uniform", "distance"])',
            p='uniform(1, 3, discrete=True)',
            metric='choices(["euclidean", "manhattan", "chebyshev", "minkowski"])'),
        1: dict(
            algorithm='ball_tree',
            n_neighbors='uniform(1, 20, discrete=True)',
            weights='choices(["uniform", "distance"])',
            leaf_size='uniform(1, 100, discrete=True)',
            p='uniform(1, 3, discrete=True)',
            metric='choices(["euclidean", "manhattan", "chebyshev", "minkowski"])'),
        2: dict(
            algorithm='kd_tree',
            n_neighbors='uniform(1, 20, discrete=True)',
            weights='choices(["uniform", "distance"])',
            leaf_size='uniform(1, 100, discrete=True)',
            p='uniform(1, 3, discrete=True)',
            metric='choices(["euclidean", "manhattan", "chebyshev", "minkowski"])')
    },
    SVC={
        0: dict(
            kernel='linear',
            C='loguniform(1e-3, 1e1)'),
        1: dict(
            kernel='poly',
            gamma='loguniform(1e-4, 1e1)', C='loguniform(1e-3, 1e1)',
            coef0='uniform(-1, 1)', degree='uniform(2, 5, discrete=True)'),
        2: dict(
            kernel='rbf',
            gamma='loguniform(1e-4, 1e1)', C='loguniform(1e-3, 1e1)'),
        3: dict(
            kernel='sigmoid',
            gamma='loguniform(1e-4, 1e1)', C='loguniform(1e-3, 1e1)',
            coef0='uniform(-1, 1)'),
    },
    GaussianProcessClassifier={
        0: dict(
            length_scale='loguniform(1-e3, 1e3)')
    },
    DecisionTreeClassifier={
        0: dict(
            max_depth='uniform(3, 10, discrete=True)',
            min_samples_leaf='uniform(0, 1)')
    },
    RandomForestClassifier={
        0: dict(
            max_depth='uniform(3, 10, discrete=True)',
            n_estimators='loguniform(1, 1000, discrete=True)',
            min_samples_leaf='uniform(0, 1)',
            max_features='choices(["sqrt", "log2", None])')
    },
    RidgeClassifier={
        0: dict(
            alpha='loguniform(1e-2, 1e1)',
            solver='choices(["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])'),
    },
    LogisticRegression={
        0: dict(
            penalty='choices(["l1", "l2"])',
            solver='choices(["liblinear", "sag", "saga"])')
    },
    Perceptron={
        0: dict(
            penalty='choices(["l1", "l2", "elasticnet", None])',
            alpha='loguniform(1e-5, 1)'),
    },
    # TODO: Probably too large for the complexity of the problems.
    MLPClassifier={
        0: dict(
            solver='sgd',
            hidden_layer_sizes='loguniform(10, 100, discrete=True, shape=1)', # 2-layers
            activation='choices(["identity", "logistic", "tanh", "relu"])',
            momentum='uniform(0.8, 0.9999)',
            learning_rate='choices(["constant", "adaptive"])',
            learning_rate_init='loguniform(1e-5, 1)',
            alpha='loguniform(1e-10, 1)'),
        1: dict(
            solver='sgd',
            hidden_layer_sizes='loguniform(10, 1000, discrete=True, shape=2)', # 2-layers
            activation='choices(["identity", "logistic", "tanh", "relu"])',
            momentum='uniform(0.8, 0.9999)',
            learning_rate='choices(["constant", "adaptive"])',
            learning_rate_init='loguniform(1e-5, 1)',
            alpha='loguniform(1e-10, 1)'),
        2: dict(
            solver='adam',
            hidden_layer_sizes='loguniform(10, 1000, discrete=True, shape=2)',
            activation='choices(["identity", "logistic", "tanh", "relu"])',
            beta_1='uniform(0.8, 0.9999)',
            beta_2='uniform(0.9, 0.9999)',
            learning_rate='choices(["constant", "adaptive"])',
            learning_rate_init='loguniform(1e-5, 1)',
            alpha='loguniform(1e-10, 1)')
    },
    AdaBoostClassifier={
        0: dict(
            n_estimators='loguniform(10, 100)',
            learning_rate='loguniform(1e-3, 1e1)'),
    },
    GaussianNB={
        0: {}
    },
    LinearDiscriminantAnalysis={
        0: {}
    },
    QuadraticDiscriminantAnalysis={
        0: {}
    })


def create_trial(dataset_config, model_config, configurator_config, previous_tags=None,
                 warm_start=0):
    space = build_space(model_config)
    configurator = build_hpo(space, **configurator_config)

    # Create client inside function otherwise MongoDB does not play nicely with multiprocessing
    mahler_client = mahler.Client()

    if previous_tags is not None:
        projection = {'output': 1, 'registry.status': 1}
        trials = mahler_client.find(tags=previous_tags + ['create_trial'], _return_doc=True,
                                    _projection=projection)
        trials = list(trials)
        assert len(trials) == 1, "{} gives {}".format(previous_tags, len(trials))
        previous_run = trials[0]
        if previous_run['registry']['status'] == 'Broken':
            raise SignalSuspend('Previous HPO is broken')
        elif previous_run['registry']['status'] != 'Completed':
            raise SignalInterruptTask('Previous HPO not completed')

        # TODO: For new hyper-parameters, sample a value from prior, even if we know the default
        #       We should compare with one setting the default.
        # TODO: For missing hyper-parameters, just drop it from params.
        for trial in previous_run['output']['trials'][:warm_start]:
            try:
                configurator.observe([trial])
            except AssertionError:
                pass

        print('There was {} compatible trials out of {}'.format(
            len(configurator.trials), len(previous_run['output']['trials'])))

        # And then we increment max_trials otherwise the configurator would already return
        # is_completed() -> True
        configurator.max_trials += len(configurator.trials)

    objectives = []
    trials = []

    numpy.random.seed(dataset_config['fold'])
    seeds = numpy.random.randint(1, 1000000, size=configurator.max_trials)

    task = mahler_client.get_current_task()
    tags = [tag for tag in task.tags if tag != task.name]
    container = task.container

    n_broken = 0
    projection = {'output': 1, 'arguments': 1, 'registry.status': 1}
    trials = mahler_client.find(tags=tags + [train.name, 'hpo'], _return_doc=True,
                                _projection=projection)

    n_uncompleted = 0
    n_trials = 0

    # NOTE: Only support sequential for now. Much simpler.
    trials = []
    print('---')
    print("Training configurator")
    for trial in trials:
        n_trials += 1
        trial = convert_mahler_task_to_trial(trial)
        if trial['status'] == 'Cancelled':
            continue

        completed_but_broken = trial['status'] == 'Completed' and not trial['objective']
        # Broken
        if trial['status'] == 'Broken' or completed_but_broken:
            n_broken += 1
        # Uncompleted
        elif trial['status'] != 'Completed':
            n_uncompleted += 1
        # Completed
        else:
            configurator.observe([trial])
            objectives.append(trial['objective'])
            trials.append(dict(params=get_params(trial['arguments'], space),
                               objective=trial['objective']))

    print('---')
    print('There is {} trials'.format(n_trials))
    print('{} uncompleted'.format(n_uncompleted))
    print('{} completed'.format(len(configurator.trials)))
    print('{} broken'.format(n_broken))

    if n_broken > 0:
        message = (
            '{} trials are broken. Suspending creation of trials until investigation '
            'is done.').format(n_broken)

        mahler_client.close()
        raise SignalSuspend(message)

    if configurator.is_completed():
        mahler_client.close()
        return dict(trials=trials, objectives=objectives)

    print('---')
    print("Generating new configurations")
    if n_uncompleted == 0:
        # NOTE: Only completed trials are in configurator.trials
        seed = int(seeds[len(configurator.trials)])
        random.seed(seed)
        params = configurator.get_params(seed=seed)
        mahler_client.register(
            train.delay(model_config=merge(model_config, params),
                        dataset_config=dataset_config, seed=seed))
    else:
        print('A trial is pending, waiting for its completion before creating a new trial.')

    mahler_client.close()
    raise SignalInterruptTask('HPO not completed')


def get_params(arguments, space):
    params = dict()
    flattened_arguments = flatten(arguments)
    for key in space.keys():
        params[key] = flattened_arguments[key]

    return unflatten(params)


def train(model_config, dataset_config, seed):
    data = build_data(**dataset_config)
    model = build_skdemocracy(seed=seed, **model_config)
    fit(model, data['train']['features'], data['train']['labels'])

    rval = dict(
        train=compute_error_rate(model, data['train']['features'], data['train']['labels']),
        valid=compute_error_rate(model, data['valid']['features'], data['valid']['labels']),
        test=compute_error_rate(model, data['test']['features'], data['test']['labels']))

    return rval


def build_model(class_name, **kwargs):
    if class_name == "GaussianProcessClassifier":
        return GaussianProcessClassifier(RBF(**kwargs))
    else:
        return globals()[class_name](**kwargs)


def build_skdemocracy(seed, **configs):
    weights = []
    classifiers = []
    rng = numpy.random.RandomState(seed)
    for class_name, model_configs in configs.items():
        for i, model_config in model_configs.items():
            if model_config['weight'] == 0:
                continue
            weights.append(model_config.pop('weight'))
            random_state = rng.randint(1, 1000000)
            classifiers.append(build_model(class_name, random_state=random_state, **model_config))

    return dict(classifiers=classifiers, weights=weights)


def fit(model, features, labels):
    for classifier in model['classifiers'].items():
        classifier.fit(features, labels)


def compute_error_rate(model, features, labels):
    preds = None
    for weight, classifier in zip(model['weights'], model['classifiers']):
        log_proba = classifier.predict_log_proba(features)
        if preds is None:
            preds = log_proba * weight
        else:
            preds += weight * log_proba

    return (numpy.argmax(preds, axis=1) != labels).mean()


def build(datasets=None, dataset_folds=None, processings=None, scenarios=None, previous_tags=None,
          warm_start=None, **kwargs):
    return LIBSVMBenchmark(datasets, dataset_folds, processings, scenarios, previous_tags, warm_start)
