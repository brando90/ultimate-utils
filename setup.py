"""
conda create -n uutils_env python=3.10
conda activate uutils_env
conda remove --all --name uutils_env
rm -rf /Users/brando/anaconda3/envs/uutils_env

pip install -e ~/ultimate-utils/ultimate-utils-proj-src/
pip install ultimate-utils

To test it:
python -c "import uutils; uutils.hello()"
python -c "import uutils; uutils.torch_uu.gpu_test()"

For full details see
```
~/ultimate-utils/tutorials_for_myself/pushing_to_pypi/README.md
```
For quick push do:
```bash
# change library version
cd ~/ultimate-utils/
rm -rf build
rm -rf dist

cd ~/ultimate-utils/
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*

cd ~/ultimate-utils/
rm -rf build
rm -rf dist
```

refs:
    - setup tools: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#using-find-or-find-packages
    - https://stackoverflow.com/questions/70295885/how-does-one-install-pytorch-and-related-tools-from-within-the-setup-py-install
"""
from setuptools import setup
from setuptools import find_packages
import os

# import pathlib

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

print('WARNING: YOU HAVE TO INSTALL PYTORCH ON YOUR OWN (WITH CUDA IF YOU NEED A GPU)')

setup(
    name='ultimate-utils',  # project name
    version='0.7.0',
    description="Brando's ultimate utils for science, machine learning and AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/brando90/ultimate-utils',
    author='Brando Miranda',
    author_email='brandojazz@gmail.com',
    python_requires='>=3.9.0',
    license='Apache-2.0',

    # currently
    package_dir={'': 'ultimate-utils-proj-src'},
    packages=find_packages('ultimate-utils-proj-src'),  # imports all modules/folders with  __init__.py & python files

    # for pytorch see doc string at the top of file
    install_requires=[
        'dill',
        'networkx>=2.5',
        'scipy',
        'scikit-learn',
        'lark-parser',
        'tensorboard',
        'pandas',
        'progressbar2',
        'transformers',
        'datasets',
        'requests',
        'aiohttp',
        'numpy',
        'plotly',
        'wandb',
        'matplotlib',
        # 'statsmodels'
        # 'statsmodels==0.12.2'
        # 'statsmodels==0.13.5'
        # - later check why we are not installing it...
        # 'seaborn'
        # 'nltk'
        'wandb',
        'twine',

        'torch',
        'torchvision',
        'torchaudio',

        # 'fairseq',

        #  - dq edge new imports
        # pip install -e ~/the-data-quality-edge-for-top-llms
        'trl',
        'transformers', # already above
        'accelerate',
        'peft',

        'datasets',
        'bitsandbytes',
        'einops',
    ]
)

