# Ultimate-utils

Hi. 
I am a collection of useful code that Brando has collected through the years that has been useful accross his projects.

To use me in development mode do (google it if you don't know what it is):

```
conda develop <PATH TO uutils/SETUP.py>
```

or 

```
pip install -e <PATH TO uutils/SETUP.py>
```

depending which package manager your using.

## How modules are imported in a python project (not package, project)

First, what is the difference between package and project? I am not sure if there is a rigurous formal definition but from reading tutorials:

- project: the folder where `setup.py` is the the rest of the packages & modules are.
- module: the namespace with collection of values (e.g. functions, constants, classes, etc). Usually a single file `module.py` or a folder with a `__init__.py` (usually called a package & it can have other modules in it).
- package: a module (single namespace) with other modules & usually other packages.
- python file vs module: no difference but TODO in more detail.

Say we have the following project we want to package:

```
(base) brandBrandoParetoopareto~/ultimate-utils $ tree packaging_project/
packaging_project/
├── module_sys_path.py
├── pkg1
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   ├── imported_module.cpython-37.pyc
│   │   ├── imported_module_by_front_dot.cpython-37.pyc
│   │   └── module.cpython-37.pyc
│   ├── imported_module.py
│   ├── imported_module_by_front_dot.py
│   └── module.py
└── setup.py
```

"packaging_project" is the name of the project name so in the `setup` function in `setup.py` I usually name the named argument name with the project name (TODO what that variable does in more depth).

Now in the root folder of the project (in thius case in `packaging_project`) we put the `setup.py` python file:

```
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
```

An important function to understand is `find_packages()` (read: http://code.nabla.net/doc/setuptools/api/setuptools/setuptools.find_packages.html). 
This function will go through every python package & modules and make them available for import statements in the given directory (& excluding whatever you tell it).
Note that this means that if you put `__init__.py` the the project that you will be able to access it as `import __init__.py` since it's a python file but NOT through `packaging_project` (if you conda develop at this project level, maybe it would work if there is a parent folder that is a project...).

Now go to the root of the project and install it in devoplment mode (with coda):

```
conda develop .
```

that will add it to you `sys.path`. Check it out with:

```
python -c "import sys; [print(p) for p in sys.path]"
```

something like this should show up:

```
(base) brandBrandoParetoopareto~/ultimate-utils $ python -c "import sys; [print(p) for p in sys.path]"

/Users/brandBrandoParetoopareto/anaconda3/envs/automl/lib/python37.zip
/Users/brandBrandoParetoopareto/anaconda3/envs/automl/lib/python3.7
/Users/brandBrandoParetoopareto/anaconda3/envs/automl/lib/python3.7/lib-dynload
/Users/brandBrandoParetoopareto/anaconda3/envs/automl/lib/python3.7/site-packages
/Users/brandBrandoParetoopareto/automl-meta-learning/automl
/Users/brandBrandoParetoopareto/higher
/Users/brandBrandoParetoopareto/ultimate-utils/uutils
/Users/brandBrandoParetoopareto/ultimate-utils/packaging_project
```

importantly the last line was added. 
Note that sometimes I've noticed the current directory being added to as the first line.

Source: 
- https://packaging.python.org/tutorials/packaging-projects/
- https://packaging.python.org/tutorials/packaging-projects/
- http://code.nabla.net/doc/setuptools/api/setuptools/setuptools.find_packages.html

### Resources

- packaging projects in python: https://packaging.python.org/tutorials/packaging-projects/
    - find_packages: http://code.nabla.net/doc/setuptools/api/setuptools/setuptools.find_packages.html
- python modules, pkgs tutorial: https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html
- Conda develop installing, unistalling & checking if it installed: https://stackoverflow.com/questions/59903548/how-does-one-check-if-conda-develop-installed-my-project-packages/59903590#59903590
- Python sys.path: https://bic-berkeley.github.io/psych-214-fall-2016/sys_path.html
- TODO:, tutorials by corey:
    - How I Manage Multiple Projects: https://www.youtube.com/watch?v=cY2NXB_Tqq0&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=16
    - pip: https://www.youtube.com/watch?v=U2ZN104hIcc&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=13
    - virtual envs: https://www.youtube.com/watch?v=N5vscPTWKOk&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=14
    - conda: https://www.youtube.com/watch?v=YJC6ldI3hWk&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=15