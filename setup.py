#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2017


'''
Setup script for the TAP-k module.
'''


import os

from setuptools import setup

from tapk import __version__


# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='TAP-k',
    version=__version__,
    description='Threshold Average Precision: '
                'metric for evaluating retrieval rankings',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/OntoGene/TAP-k',
    author='Lenz Furrer',
    author_email='furrer@cl.uzh.ch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3',
    py_modules=['tapk'],
    entry_points={
        'console_scripts': [
            'TAP-k=tapk:main',
        ],
    },
)
