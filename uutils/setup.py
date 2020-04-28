from setuptools import setup
from setuptools import find_packages

setup(
    name='uutils', # project name
    version='0.1.0',
    description='Brandos utils for science',
    #url
    author='Brando Miranda',
    author_email='miranda9@illinois.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=['torch','numpy','scikit-learn','scipy','matplotlib','pyyml','torchviz','tensorboard',
    'graphviz','torchvision','matplotlib']
)

#install_requires=['numpy>=1.11.0']
