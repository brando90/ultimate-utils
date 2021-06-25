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
    packages=find_packages(), # imports all modules (folder with __init__.py) & python files in this folder (since defualt args are . and empty exculde i.e. () )
    install_requires=['dill', 'torch', 'pandas', 'pygraphviz', 'lark'
                      'pydot']
)

#install_requires=['numpy>=1.11.0']
