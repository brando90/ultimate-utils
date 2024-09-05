"""
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
    - The document explains setuptools package discovery for correct python package creation: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#using-find-or-find-packages
        - note:  setuptools: is a Python library designed to facilitate the packaging, distribution, and installation of Python projects
"""
from setuptools import setup
from setuptools import find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

print('WARNING: YOU HAVE TO INSTALL PYTORCH ON YOUR OWN (WITH CUDA IF YOU NEED A GPU)')

setup(
    name='ultimate-utils',  # project name
    version='0.8.0',
    description="Brando's Ultimate Utils for Science, Machine Learning, and AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/brando90/ultimate-utils',
    author='Brando Miranda',
    author_email='brandojazz@gmail.com',
    python_requires='>=3.10.0',
    license='Apache-2.0',

    # ref: https://chat.openai.com/c/d0edae00-0eb2-4837-b492-df1d595b6cab
    # The `package_dir` parameter is a dictionary that maps package names to directories.
    # A key of an empty string represents the root package, and its corresponding value
    # is the directory containing the root package. Here, the root package is set to the
    # 'src' directory.
    #
    # The use of an empty string `''` as a key is significant. In the context of setuptools,
    # an empty string `''` denotes the root package of the project. It means that the
    # packages and modules located in the specified directory ('src' in this case) are
    # considered to be in the root of the package hierarchy. This is crucial for correctly
    # resolving package and module imports when the project is installed.
    #
    # By specifying `{'': 'src'}`, we are informing setuptools that the 'src' directory is
    # the location of the root package, and it should look in this directory to find the
    # Python packages and modules to be included in the distribution.
    package_dir={'': 'py_src'},

    # The `packages` parameter lists all Python packages that should be included in the
    # distribution. A Python package is a way of organizing related Python modules into a
    # directory hierarchy. Any directory containing an __init__.py file is considered a
    # Python package.
    #
    # `find_packages('src')` is a convenience function provided by setuptools, which
    # automatically discovers and lists all packages in the specified 'src' directory.
    # This means it will include all directories in 'src' that contain an __init__.py file,
    # treating them as Python packages to be included in the distribution.
    #
    # By using `find_packages('src')`, we ensure that all valid Python packages inside the
    # 'src' directory, regardless of their depth in the directory hierarchy, are included
    # in the distribution, eliminating the need to manually list them. This is particularly
    # useful for projects with a large number of packages and subpackages, as it reduces
    # the risk of omitting packages from the distribution.
    packages=find_packages('py_src'),
    # When using `pip install -e .`, the package is installed in 'editable' or 'develop' mode.
    # This means that changes to the source files immediately affect the installed package
    # without requiring a reinstall. This is extremely useful during development as it allows
    # for testing and iteration without the constant need for reinstallation.
    #
    # In 'editable' mode, the correct resolution of package and module locations is crucial.
    # The `package_dir` and `packages` configurations play a vital role in this. If the
    # `package_dir` is incorrectly set, or if a package is omitted from the `packages` list,
    # it can lead to ImportError due to Python not being able to locate the packages and
    # modules correctly.
    #
    # Therefore, when using `pip install -e .`, it is essential to ensure that `package_dir`
    # correctly maps to the root of the package hierarchy and that `packages` includes all
    # the necessary packages by using `find_packages`, especially when the project has a
    # complex structure with nested packages. This ensures that the Python interpreter can
    # correctly resolve imports and locate the source files, allowing for a smooth and
    # efficient development workflow.

   # for pytorch see doc string at the top of file
    install_requires=[
        'fire',
        'dill',
        # 'networkx>=2.5',
        'scipy',
        'scikit-learn',
        'lark-parser',
        'tensorboard',
        'pandas',
        'progressbar2',
        'requests',
        'aiohttp',
        'numpy',
        'plotly',
        'wandb',
        'matplotlib',
        'nvidia-htop',
        'openai',
        'anthropic',
        'jsonlines',
        # 'statsmodels'
        # 'statsmodels==0.12.2'
        # 'statsmodels==0.13.5'
        # - later check why we are not installing it...
        'seaborn',
        # 'nltk'
        'twine',
        'dspy-ai',
        'ragatouille',
        'torch',  # here so it's there for default but if using vllm see bellow or readme.md
        # 'torchvision',
        # 'torchaudio',
        'trl',
        'transformers',
        'peft',
        'accelerate',
        'datasets',
        'bitsandbytes',
        'evaluate',
        'einops',
        'sentencepiece', # needed llama2 tokenizer
        # 'zstandard', # needed for eval of all the pile

        # def does not work for mac
        # # -- ref: https://github.com/vllm-project/vllm/issues/2747 
        # pip install torch==2.2.1
        # pip install vllm==0.4.1
        # 'torch==2.2.1',
        # 'vllm==0.4.1', 
        # # --

        # # mercury: https://github.com/vllm-project/vllm/issues/2747
        # 'dspy-ai',
        # # 'torch==2.1.2+cu118',  # 2.2 net supported due to vllm see: https://github.com/vllm-project/vllm/issues/2747
        # 'torch==2.2.2',  # 2.2 net supported due to vllm see: https://github.com/vllm-project/vllm/issues/2747
        # # 'torchvision',
        # # 'torchaudio',
        # # 'trl',
        # 'transformers',
        # 'accelerate',
        # # 'peft',
        # # 'datasets==2.18.0', 
        # 'datasets',  
        # 'evaluate', 
        # 'bitsandbytes',
        # # 'einops',
        # # 'vllm==0.4.0.post1', # my gold-ai-olympiad project uses 0.4.0.post1 ref: https://github.com/vllm-project/vllm/issues/2747

        # # ampere
        # 'dspy-ai',
        # # 'torch==2.1.2+cu118',  # 2.2 not supported due to vllm see: https://github.com/vllm-project/vllm/issues/2747
        # # 'torch==2.1.2',  # 2.2 not supported due to vllm see: https://github.com/vllm-project/vllm/issues/2747
        # # 'torch==2.2.1',  # 2.2 not supported due to vllm see: https://github.com/vllm-project/vllm/issues/2747
        # 'torch==2.2.1',  # 2.2 not supported due to vllm see: https://github.com/vllm-project/vllm/issues/2747
        # # 'torchvision',
        # # 'torchaudio',
        # # 'trl',
        # # 'transformers==4.39.2',
        # 'transformers>=4.40',
        # 'accelerate==0.29.2',
        # # 'peft',
        # # 'datasets==2.18.0', 
        # 'datasets==2.14.7',  
        # 'evaluate==0.4.1', 
        # 'bitsandbytes== 0.43.0',
        # 'einops',
        # 'flash-attn>=2.5.8',
        # 'vllm==0.4.1', # my gold-ai-olympiad project uses 0.4.0.post1 ref: https://github.com/vllm-project/vllm/issues/2747
        # # pip install -q -U google-generativeai
    ]
)

