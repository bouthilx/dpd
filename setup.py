from os import path
from io import open
from setuptools import setup, find_namespace_packages

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
    install_requires=['torch', 'orion.core'],   # 'kleio.core'],
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False
)