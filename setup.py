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
    author_email='Xavier Bouthillier, Simon Guiroy, Cesar Laurent',
    package_dir={'': 'src'},
    packages=find_namespace_packages('src'),
    install_requires=[],
    extras_require = {
        'execute':  [
            'torch==1.0.0', 'torchvision', 'torchnet', 'pytorch-ignite', 'tqdm',
            'h5py==2.9.0', 'Pillow==5.3.0', 'filelock==3.0.10'],
        'configure': [
            'orion.core==db2d068',
            'orion.algo.skopt==577aabf'],
        'deploy': [
            'kleio.core==0.1.0.a',
            'flow==6354eff']
    },
    dependency_links=[
        "git+https://github.com/bouthilx/orion.git@db2d068#egg=orion.core-db2d068",
        "git+https://gitlab.com/bouthilx/orion.algo.skopt.git@577aabf#egg=orion.algo.skopt-577aabf",
        "git+https://github.com/epistimio/kleio.git@prototype#egg=kleio.core-0.1.0.a",
        "git+https://github.com/bouthilx/flow.git@6354eff#egg=flow-6354eff",
    ],
    setup_requires=['setuptools', 'pytest-runner'],
    tests_require=['pytest'],
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False
)
