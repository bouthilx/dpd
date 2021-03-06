#!/usr/bin/env python
from io import open
from os import path

from setuptools import find_namespace_packages, setup


HERE = path.abspath(path.dirname(__file__))


with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read().strip()


setup(
    name='repro',
    description='',
    long_description=LONG_DESCRIPTION,
    author_email='Xavier Bouthillier, Cesar Laurent, Simon Guiroy',
    package_dir={'': 'src'},
    packages=find_namespace_packages('src'),
    install_requires=[
        'orion.core',
        'orion.algo.skopt', 'tqdm', 'dataclasses', 'matplotlib'],
    extras_require = {
        'coco': [
            'cocoex'],
        'libsvm': [
            'lxml'
            ],
        'mini': [
            'torch==1.0.0', 'torchvision', 'torchnet', 'pytorch-ignite',
            'filelock==3.0.10', 'h5py==2.9.0', 'Pillow==5.3.0'],
        'bayesopt': [
            'orion.algo.skopt'
            ],
        'tpe': [
            'orion.algo.optuna'
            ],
        'dpd': [
            'numpy', 'scipy',
            ],
        'configure': [
            'mahler.registry.mongodb',
            'orion.core==db2d068',
            'orion.algo.skopt==577aabf'],
        'deploy': [
            'mahler.core==83438f1',
            'mahler.registry.mongodb==87146aa',
            'mahler.scheduler.flow==b6d2d0e',
            'flow==6354eff'],
        'monitor': [
            'tqdm==4.28.1',
            'numpy==1.15.4',
            'dash==0.32.2',
            'dash-core-components==0.41.0',
            'dash-html-components==0.13.2',
            'flask-caching==1.4.0'
            ],
    },
    dependency_links=[
        "git+https://github.com/bouthilx/mahler.git@83438f1#egg=mahler.core-83438f1",
        "git+https://github.com/bouthilx/mahler.registry.mongodb.git@87146aa#egg=mahler.registry.mongodb-87146aa",
        "git+https://github.com/bouthilx/mahler.scheduler.flow.git@b6d2d0e#egg=mahler.scheduler.flow-b6d2d0e",
        "git+https://github.com/bouthilx/orion.git@db2d068#egg=orion.core-db2d068",
        "git+https://github.com/Epistimio/orion.algo.skopt.git@71dcdd8#egg=orion.algo.skopt-71dcdd8",
        "git+https://github.com/bouthilx/flow.git@6354eff#egg=flow-6354eff",
    ],
    setup_requires=['setuptools', 'pytest-runner'],
    tests_require=['pytest'],
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False
)
