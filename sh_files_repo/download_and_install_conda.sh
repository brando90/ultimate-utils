# RUN ALL THE INSTRUCTIONS! PLEASE!
echo $HOME
# -- Install miniconda
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash $HOME/miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
# - Set up conda
conda init
# conda init zsh
conda init bash
conda install conda-build
conda update -n base -c defaults conda
conda update conda
# - Create conda env
conda create -n my_env python=3.10
conda activate my_env
## conda remove --name my_env --all
# - Make sure pip is up to date
which python
pip install --upgrade pip
pip3 install --upgrade pip
which pip
which pip3

# - [NOT This one, full anacoda is huge] installing full anaconda
#wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh
#wget https://repo.continuum.io/conda/Anaconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh
#nohup bash ~/anaconda.sh -b -p $HOME/anaconda > anaconda_install.out &
#ls -lah $HOME | grep anaconda
#source ~/anaconda/bin/activate
