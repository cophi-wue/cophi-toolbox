#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

PROJECT = 'Computer Philology Toolbox'
VERSION = "0.1"
REVISION = "0.1.1.dev0"
AUTHOR = "DARIAH-DE Wuerzburg Group"
AUTHOR_EMAIL = "pielstroem@biozentrum.uni-wuerzburg.de"

setup(
    name='cophi_toolbox',
    version=REVISION,
    description=PROJECT,
    # url
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    # license
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    # keywords
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=0.19.2',
        'regex>=2017.01.14',
        'numpy>=1.3',
        'lxml>=3.6.4'
    ],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', PROJECT),
            'version': ('setup.py', VERSION),
            'release': ('setup.py', REVISION),
        }
    }
)
