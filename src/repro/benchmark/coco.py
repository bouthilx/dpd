import itertools
import logging
import random

import numpy

from orion.core.io.space_builder import Space, DimensionBuilder

from repro.hpo.base import build_hpo

try:
    import cocoex
except ImportError:
    cocoex = None


try:
    import mahler.client as mahler
    from mahler.core.utils.errors import SignalInterruptTask, SignalSuspend
except ImportError:
    mahler = None


logger = logging.getLogger(__name__)


class COCOBenchmark:

    name = 'coco'

    def __init__(self, problem_ids=None, dimensions=None, instances=None, scenarios=None,
                 previous_tags=None, warm_start=0):
        self.verify(problem_ids, dimensions, instances, scenarios)
        self._problem_ids = problem_ids
        self._dimensions = dimensions
        self._instances = instances
        self._scenarios = scenarios
        self._previous_tags = previous_tags
        self._warm_start = warm_start
        self.suite = cocoex.Suite("bbob", "year: 2016", "")

    def verify(self, problems, dimensions, instances, scenarios):
        # TODO
        pass

    def add_subparser(self, subparsers):
        benchmark_parser = subparsers.add_parser('coco')

        benchmark_parser.add_argument('--problems', choices=self.problem_ids, type=int, nargs='*')
        benchmark_parser.add_argument('--dimensions', choices=self.dimensions, type=int, nargs='*')
        benchmark_parser.add_argument('--instances', choices=self.instances, type=int, nargs='*')
        benchmark_parser.add_argument('--scenarios', choices=self.scenarios, type=str, nargs='*')
        benchmark_parser.add_argument('--warm-start', type=int, default=50)
        benchmark_parser.add_argument('--max-trials', type=int, default=50)

        return benchmark_parser

    @property
    def problem_ids(self):
        if getattr(self, '_problem_ids', None):
            return self._problem_ids

        return list(range(1, 25))

    @property
    def dimensions(self):
        if self._dimensions:
            return self._dimensions

        return self.suite.dimensions

    @property
    def instances(self):
        if self._instances:
            return self._instances

        return list(range(1, 6))

    @property
    def scenarios(self):
        if getattr(self, '_scenarios', None):
            return self._scenarios

        # TODO: Define how to set 2.1, 2.2 and 2.3.
        return ['0.0',  # No diff
                '2.1',  # Fewer H-Ps
                '2.2',  # More H-Ps
                '2.3.a',  # Prior changed
                '2.3.b',  # Prior changed
                '2.4.a',
                '2.4.b',
                '3.1',  # Code change without any effect
                '3.2',  # Loss is reversed
                '3.3',  # Loss is scaled
                '3.4',  # H-P is reversed
                '3.5']  # H-P is scaled

    @property
    def problems(self):
        configs = itertools.product(self.problem_ids, self.dimensions, self.instances,
                                    self.scenarios)

        for config in configs:
            try:
                self.suite.get_problem_by_function_dimension_instance(*config[:-1])
            except cocoex.exceptions.NoSuchProblemException:
                continue
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
        tags = ['coco',
                'f{:03d}'.format(self.id), 'd{:03d}'.format(self.dimension),
                'i{:02d}'.format(self.instance_id),
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
        problem = build_problem(**self.config)
        space_config = dict()
        for dim in range(problem.dimension):
            name = get_dim_name(dim)

            lower = problem.lower_bounds[dim]
            upper = problem.upper_bounds[dim]

            if self.scenario == "2.3.a":
                space_config[name] = dict(lower_bound=lower, upper_bound=(lower + upper) * 3. / 4.)
            elif self.scenario == "2.3.b":
                space_config[name] = dict(lower_bound=(lower + upper) * 1. / 4., upper_bound=upper)
            elif self.scenario == "2.4.a":
                space_config[name] = dict(lower_bound=lower, upper_bound=(lower + upper) / 2.)
            elif self.scenario == "2.4.b":
                space_config[name] = dict(lower_bound=(lower + upper) / 2., upper_bound=upper)

        return space_config

    @property
    def config(self):
        return dict(
            problem_id=self.id,
            nb_of_dimensions=self.dimension,
            instance_id=self.instance_id)

    def execute(self, configurator_config):
        rval = hpo_coco(
            self.config,
            self.space_config,
            configurator_config=configurator_config,
            previous_tags=self.previous_tags,
            warm_start=self.warm_start)

        print(self.id, self.dimension, self.instance_id)
        print(len(rval['objectives']), min(rval['objectives']))

    def register(self, mahler_client, configurator_config, container, tags):

        # import pdb
        # pdb.set_trace()

        mahler_client.register(
            hpo_coco.delay(
                problem_config=self.config,
                space_config=self.space_config,
                configurator_config=configurator_config,
                previous_tags=self.previous_tags),
            container=container, tags=tags)

        print('Registered', *tags)

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


# function_id = 1
# dimensions = 2
# instance_id = 1


DIM_NAME_TEMPLATE = 'dim-{:03d}'


def get_dim_name(dim_id):
    return DIM_NAME_TEMPLATE.format(dim_id)


def build_problem(problem_id, nb_of_dimensions, instance_id):
    suite = cocoex.Suite("bbob", "year: 2016", "")
    return suite.get_problem_by_function_dimension_instance(problem_id, nb_of_dimensions,
                                                            instance_id)


# With mahler:
# Check if previous exists, otherwise, register it as well
# NOTE: previous might not be done, register new anyway and it will only create trials when previous
# is done.

# Scenarios
# 0. No diff
# 1. -
# 2. H-Ps
#     2.1 fewer H-Ps
#     2.2 More H-Ps
#     2.3 Prior changed with half overlap
#     2.4 Prior changed with no overlap
# 3. Code change
#     3.1 Without any effect
#     3.2 Loss is reversed
#     3.3 Loss is scaled
#     3.4 HP is reversed
#     3.5 HP is scaled
#     3.6 -


# 0.o o -> (lower, upper)
# 0.  (lower, upper) -> (lower, upper)
# 2.4.a.o o -> (mid, upper)
# 2.4.a (lower, mid) -> (mid, upper)
# 2.4.b.o o -> (lower, mid)
# 2.4.b (mid, upper) -> (lower, mid)


def hpo_coco(problem_config, space_config, configurator_config, previous_tags=None, warm_start=0):

    problem = build_problem(**problem_config)
    space = build_space(problem, **space_config)
    configurator = build_hpo(space, **configurator_config)

    if previous_tags is not None:
        # Create client inside function otherwise MongoDB does not play nicely with multiprocessing
        mahler_client = mahler.Client()
        projection = {'output': 1, 'registry.status': 1}
        trials = mahler_client.find(tags=previous_tags, _return_doc=True,
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

    numpy.random.seed(problem_config['instance_id'])
    seeds = numpy.random.randint(1, 1000000, size=configurator.max_trials)

    while not configurator.is_completed():
        seed = seeds[len(configurator.trials)]
        random.seed(seed)
        params = configurator.get_params(seed=seed)
        rval = problem([params[get_dim_name(dim)] for dim in range(problem.dimension)])
        trial = dict(params=params, objective=rval)
        trials.append(trial)
        configurator.observe([trial])
        objectives.append(rval)

    return dict(trials=trials, objectives=objectives)


def build_space(problem, **space_config):
    space = Space()
    for dim in range(problem.dimension):
        name = get_dim_name(dim)

        if space_config.get(name, {}) is None:
            continue

        config = dict(
            prior='uniform', lower_bound=problem.lower_bounds[dim],
            upper_bound=problem.upper_bounds[dim])
        config.update(space_config.get(name, {}))

        space[name] = DimensionBuilder().build(
            name, '{prior}({lower_bound}, {upper_bound})'.format(**config))

    print(space)

    return space


if mahler is not None:
    hpo_coco = mahler.operator(resources={'cpu': 2, 'mem': '20MB'}, resumable=False)(hpo_coco)


if cocoex is not None:
    def build(problems=None, dimensions=None, instances=None, scenarios=None, previous_tags=None,
              warm_start=None, **kwargs):
        return COCOBenchmark(problems, dimensions, instances, scenarios, previous_tags, warm_start)
