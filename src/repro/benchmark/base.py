from repro.utils.factory import fetch_factories


factories = fetch_factories('repro.benchmark', __file__)


def build_benchmark(name, **kwargs):
    return factories[name](**kwargs)


def build_problem(name, config):
    return factories[name]().build(**config)


def build_benchmark_subparsers(subparsers):
    benchmark_subparsers = []
    for name in factories.keys():
        benchmark = factories[name]()
        benchmark_subparsers.append(benchmark.add_subparser(subparsers))

    return benchmark_subparsers


"""python src/repro/cli/hpot.py execute coco --tags v7.1.0"""
"""python src/repro/cli/hpot.py register coco --tags v7.1.0"""
