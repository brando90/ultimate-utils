# - optional: delete conda
#conda remove -n uutils_env --all

# - create conda env
conda create -n uutils_env python=3.9
conda activate uutils_env

# - pip update wandb
pip install --upgrade wandb

# - dev install
pip install -e .