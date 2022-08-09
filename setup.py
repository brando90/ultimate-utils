"""
conda create -n uutils_env python=3.9
conda activate uutils_env
conda remove --all --name uutils_env
rm -rf /Users/brando/anaconda3/envs/uutils_env

conda create -n uutils_env_gpu python=3.9
conda activate uutils_env_gpu
conda remove --all --name uutils_env_gpu
rm -rf /Users/brando/anaconda3/envs/uutils_env_gpu

conda create -n uutils_env_cpu python=3.9
conda activate uutils_env_cpu
conda remove --all --name uutils_env_gpu
rm -rf /Users/brando/anaconda3/envs/uutils_env_gpu

pip install -e ~/ultimate-utils/ultimate-utils-proj-src/
pip install ultimate-utils

To test it:
python -c "import uutils; uutils.hello()"
python -c "import uutils; uutils.torch_uu.hello()"

python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
python -c "import uutils; uutils.torch_uu.gpu_test()"

PyTorch:
- basing the torch install from the pytorch website as of this writing: https://pytorch.org/get-started/locally/
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch torchvision torchaudio
pip install torch torchvision torchaudio

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

setup(
    name='ultimate-utils',  # project name
    version='0.5.5',
    description="Brando's ultimate utils for science, machine learning and AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/brando90/ultimate-utils',
    author='Brando Miranda',
    author_email='brandojazz@gmail.com',
    python_requires='>=3.9.0',
    license='MIT',

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
        'torchtext==0.10.1',
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
        # 'seaborn'
        # 'nltk'

        # at the end because it might fail + you might need to pip install it directly for your cuda version
        'torch>=1.4.0,<1.10.0',
        'torchvision>=0.5.0,<0.11.0',
        'torchaudio>=0.9.1,<0.10.0',
        # 'torch==1.9.1',
        # 'torchvision==0.10.1',
        # 'torchaudio==0.9.1',
    ]
)
