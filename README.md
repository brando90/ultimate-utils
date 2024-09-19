# Ultimate-utils

Ulitmate-utils (or uutils) is collection of useful code that Brando has collected through the years that has been useful accross his projects.
Mainly for machine learning and programming languages tasks.

## Installing Ultimate-utils

> WARNING: YOU HAVE TO INSTALL PYTORCH ON YOUR OWN (WITH CUDA IF YOU NEED A GPU)

# Installation [Dev]

To install with code do: 
```bash
conda create -n uutils python=3.11 -y
conda activate uutils
# conda remove --all --name uutils

pip install -e ~/ultimate-utils
```

To install with venv do:
```bash
deactivate
mkdir ~/.virtualenvs
ls ~/.virtualenvs
python3.11 -m venv ~/.virtualenvs/uutils
# python3 -m venv ~/.virtualenvs/uutils
source ~/.virtualenvs/uutils/bin/activate
pip install --upgrade pip
which python

pip install -e ~/ultimate-utils
```

To test (any) pytorch do:
```bash
python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
```

## Install vLLM

To install vllm:
```bash
# - Install vllm
# FAILED: bellow failed to install vllm with uutils first installing it with default setup.py then 
# pip install --upgrade pip
# pip install torch==2.2.1
# pip install vllm==0.4.1
# - Installed vllm on skampere1
pip install --upgrade pip
pip uninstall torchvision vllm vllm-flash-attn flash-attn xformers
pip install torch==2.2.1 vllm==0.4.1 
# fails install
# pip install flash-attn==2.6.3
```

## Pushing to pypi
For full details see
```bash
~/ultimate-utils/tutorials_for_myself/pushing_to_pypi/README.md
```
For quick push do:
```bash
cd ~/ultimate-utils/
rm -rf build
rm -rf dist
cd ~/ultimate-utils/
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
```
then:
```bash
cd ~/ultimate-utils/
rm -rf build
rm -rf dist
```