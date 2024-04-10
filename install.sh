# - optional: delete conda
#conda remove -n uutils_env --all -y
conda remove -n data_quality --all -y

# - create conda env
conda create -n uutils_env python=3.9
conda activate uutils_env

# - pip update wandb
pip install --upgrade wandb

# - dev install
pip install -e .