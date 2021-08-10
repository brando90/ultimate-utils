from setuptools import setup
from setuptools import find_packages

setup(
    name='uutils', # project name
    version='0.1.0',
    description='Brandos ultimate utils for science',
    #url
    author='Brando Miranda',
    author_email='brandojazz@gmail.com',
    license='MIT',
    packages=find_packages(), # imports all modules (folder with __init__.py) & python files in this folder (since defualt args are . and empty exculde i.e. () )
    install_requires=['dill',
                      'networkx>=2.5',
                      'scipy',
                      'scikit-learn',
                      'lark-parser',
                      'torchtext',
                      'tensorboard',
                      'pandas',


                      'pygraphviz']
)

#install_requires=['numpy>=1.11.0']
