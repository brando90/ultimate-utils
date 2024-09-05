from setuptools import setup
from setuptools import find_packages

setup(
    name='packaging_project', # project name
    version='0.1.0',
    description="Brando's sample packaging tutorial",
    #url
    author='Brando Miranda',
    author_email='miranda9@illinois.edu',
    license='MIT',
    packages=find_packages(), # default find_packages(where='.', exculde=())
    install_requires=['torch','numpy','scikit-learn','scipy','matplotlib','pyyml','torchviz','tensorboard',
    'graphviz','torchvision','matplotlib']
)

