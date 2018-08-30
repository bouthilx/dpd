#!/usr/bin/env python
from io import open
from os import path

from setuptools import find_namespace_packages, setup


HERE = path.abspath(path.dirname(__file__))


with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read().strip()


setup(
    name='sgdad',
    description='',
    long_description=LONG_DESCRIPTION,
    author_email='Xavier Bouthillier, Simon Guiroy',
    package_dir={'': 'src'},
    packages=find_namespace_packages('src'),
    install_requires=['torch', 'torchvision', 'pytorch-ignite', 'orion.core'],   # 'kleio.core'],
    setup_requires=['setuptools', 'pytest-runner'],
    tests_require=['pytest'],
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False
)
