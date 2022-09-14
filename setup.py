#!/usr/bin/env python
from os import path
import io

from setuptools import setup, find_packages

import weka

CURRENT_DIR = path.abspath(path.dirname(__file__))


def get_reqs(*fns):
    lst = []
    for fn in fns:
        with open(path.join(CURRENT_DIR, fn), encoding='utf-8') as fin:
            for package in fin.readlines():
                package = package.strip()
                if not package:
                    continue
                lst.append(package.strip())
    return lst


this_directory = path.abspath(path.dirname(__file__))
readme_fn = path.join(this_directory, 'README.md')
long_description = ''
if path.isfile(readme_fn):
    with open(readme_fn, encoding='utf-8', errors='ignore') as f:
        long_description = f.read()

setup(
    name='weka',
    version=weka.__version__,
    description='A Python wrapper for the Weka data mining library.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Chris Spencer',
    author_email='chrisspen@gmail.com',
    url='https://github.com/chrisspen/weka',
    license='LGPL License',
    packages=find_packages(),
    package_data={
        'weka': ['fixtures/*'],
    },
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: General",
    ],
    platforms=['OS Independent'],
    install_requires=get_reqs('pip-requirements.txt'),
)
