"""Legacy setuptools shim.

Canonical package metadata and the build backend now live in ``pyproject.toml``.
This file remains only for legacy tooling that still imports ``setup.py`` and
for setuptools package discovery under ``py_src``.
"""
from setuptools import find_packages, setup

setup(
    package_dir={'': 'py_src'},
    packages=find_packages('py_src'),
)
