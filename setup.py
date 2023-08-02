# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()


setup(
    name='marie',
    version='0.1.0',
    description='',
    long_description=readme,
    author='Dan Tran',
    author_email='nmdt2@cam.ac.uk',
    url='https://github.com/picas9dan/marie-llama',
    packages=find_packages(exclude=('tests'))
)